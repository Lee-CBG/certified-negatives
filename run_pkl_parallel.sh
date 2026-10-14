#!/usr/bin/env bash
# Run the 3 ablation conditions SIMULTANEOUSLY, one per GPU.
#
#   ./run_pkl_parallel.sh data/BLOSUM62.pkl
#
# The three conditions are completely independent runs -- different models,
# different weights, no shared state -- so this is embarrassingly parallel.
# Wall time becomes the time of the SLOWEST single condition rather than
# the sum of all three.
#
# Logs stream to results/log_<mode>.txt so you can tail them while they run.
set -euo pipefail

PKL="${1:-data/BLOSUM62.pkl}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
export PYTHONPATH=src

if [ ! -f "$PKL" ]; then
  echo "File not found: $PKL"
  exit 1
fi

mkdir -p results

echo "########## INSPECT ##########"
python3 src/inspect_pkl.py --pkl "$PKL"

echo
echo "########## LAUNCHING 3 CONDITIONS IN PARALLEL ##########"
nvidia-smi --query-gpu=index,name,memory.used,memory.total \
           --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"
echo

MODES=(full tcr_only epi_only)
PIDS=()

for i in "${!MODES[@]}"; do
  MODE="${MODES[$i]}"
  LOG="results/log_${MODE}.txt"
  echo "  GPU $i  ->  $MODE   (log: $LOG)"
  CUDA_VISIBLE_DEVICES="$i" python3 src/train_pkl.py \
      --pkl "$PKL" \
      --mode "$MODE" \
      --seed "$SEED" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --tf32 \
      --out "results/pkl_${MODE}.json" \
      > "$LOG" 2>&1 &
  PIDS+=($!)
done

echo
echo "Running. Watch progress with:"
echo "    tail -f results/log_full.txt"
echo "    nvidia-smi -l 5"
echo

FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  [done] ${MODES[$i]}"
  else
    echo "  [FAILED] ${MODES[$i]} -- see results/log_${MODES[$i]}.txt"
    FAILED=1
  fi
done

if [ "$FAILED" -ne 0 ]; then
  echo
  echo "At least one condition failed; not writing a report."
  exit 1
fi

echo
echo "########## REPORT ##########"
python3 src/report.py --results-dir results --prefix pkl_