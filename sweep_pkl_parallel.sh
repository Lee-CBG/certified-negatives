#!/usr/bin/env bash
# Seed sweep across all available GPUs using a simple job queue.
#
#   ./sweep_pkl_parallel.sh data/BLOSUM62.pkl
#   SEEDS="0 1 2 3 4" NGPU=4 ./sweep_pkl_parallel.sh data/BLOSUM62.pkl
#
# 5 seeds x 3 conditions = 15 runs. A queue keeps every GPU busy instead of
# running in fixed batches, so no GPU idles waiting for a slow straggler.
#
# Results: results_seed<N>/pkl_<mode>.json
# Logs:    results_seed<N>/log_<mode>.txt
set -euo pipefail

PKL="${1:-data/BLOSUM62.pkl}"
SEEDS="${SEEDS:-0 1 2 3 4}"
NGPU="${NGPU:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
NGPU="${NGPU:-1}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-30}"
export PYTHONPATH=src

if [ ! -f "$PKL" ]; then
  echo "File not found: $PKL"
  exit 1
fi

echo "Sweeping seeds [$SEEDS] across $NGPU GPU(s)"
echo

# Build the full job list
JOBS=()
for S in $SEEDS; do
  for M in full tcr_only epi_only; do
    JOBS+=("$S:$M")
  done
done
echo "Total runs: ${#JOBS[@]}"
echo

# Track which PID occupies which GPU slot
declare -A SLOT_PID

launch() {
  local gpu="$1" seed="$2" mode="$3"
  local dir="results_seed${seed}"
  mkdir -p "$dir"
  echo "  GPU $gpu  <-  seed $seed / $mode"
  CUDA_VISIBLE_DEVICES="$gpu" python3 src/train_pkl.py \
      --pkl "$PKL" --mode "$mode" --seed "$seed" \
      --epochs "$EPOCHS" --patience "$PATIENCE" --tf32 \
      --out "$dir/pkl_${mode}.json" \
      > "$dir/log_${mode}.txt" 2>&1 &
  SLOT_PID[$gpu]=$!
}

JOB_I=0

# Fill every GPU slot to start
for ((g=0; g<NGPU && JOB_I<${#JOBS[@]}; g++)); do
  IFS=':' read -r S M <<< "${JOBS[$JOB_I]}"
  launch "$g" "$S" "$M"
  JOB_I=$((JOB_I+1))
done

# As each finishes, immediately start the next queued job on that GPU
while [ "$JOB_I" -lt "${#JOBS[@]}" ] || [ "${#SLOT_PID[@]}" -gt 0 ]; do
  sleep 5
  for g in "${!SLOT_PID[@]}"; do
    pid="${SLOT_PID[$g]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || echo "  [warn] a run on GPU $g exited non-zero"
      unset 'SLOT_PID[$g]'
      if [ "$JOB_I" -lt "${#JOBS[@]}" ]; then
        IFS=':' read -r S M <<< "${JOBS[$JOB_I]}"
        launch "$g" "$S" "$M"
        JOB_I=$((JOB_I+1))
      fi
    fi
  done
done

echo
echo "########## SWEEP SUMMARY ##########"
python3 - "$SEEDS" <<'PY'
import json, os, sys, statistics
seeds = sys.argv[1].split()
diffs, macro = [], []
print(f"\n{'seed':>6}{'full':>10}{'cdr3_only':>12}{'epi_only':>11}{'contrib':>12}")
print("-" * 51)
for s in seeds:
    d = f"results_seed{s}"
    try:
        f = json.load(open(f"{d}/pkl_full.json"))
        t = json.load(open(f"{d}/pkl_tcr_only.json"))
        e = json.load(open(f"{d}/pkl_epi_only.json"))
    except FileNotFoundError:
        print(f"{s:>6}  (incomplete)")
        continue
    c = f["test_auroc"] - t["test_auroc"]
    diffs.append(c)
    macro.append(f["macro_auroc_per_epitope"] - t["macro_auroc_per_epitope"])
    print(f"{s:>6}{f['test_auroc']:>10.4f}{t['test_auroc']:>12.4f}"
          f"{e['test_auroc']:>11.4f}{c:>+12.4f}")
if diffs:
    print("-" * 51)
    print(f"\nepitope contribution: mean {statistics.mean(diffs):+.4f}"
          f"  sd {statistics.pstdev(diffs):.4f}"
          f"  range [{min(diffs):+.4f}, {max(diffs):+.4f}]")
    print(f"on macro AUROC:       mean {statistics.mean(macro):+.4f}"
          f"  sd {statistics.pstdev(macro):.4f}")
PY