#!/usr/bin/env bash
# Run the 3-condition ablation on a pre-encoded BLOSUM62 pkl.
#
#   ./run_pkl.sh data/BLOSUM62.pkl
#
# Results go to results/pkl_*.json
# Uses catELMo's architecture: Dense 2048 per branch, Dense 1024 head.
# Early stopping: patience 30, max 200 epochs (same as catELMo's bap.py).
set -euo pipefail

PKL="${1:-data/BLOSUM62.pkl}"
SEED="${SEED:-0}"
export PYTHONPATH=src

if [ ! -f "$PKL" ]; then
  echo "File not found: $PKL"
  echo "Usage: ./run_pkl.sh path/to/BLOSUM62.pkl"
  exit 1
fi

mkdir -p results

echo "########## INSPECT ##########"
python3 src/inspect_pkl.py --pkl "$PKL"

for MODE in full tcr_only epi_only; do
  echo
  echo "########## $MODE ##########"
  python3 src/train_pkl.py \
    --pkl "$PKL" \
    --mode "$MODE" \
    --seed "$SEED" \
    --out "results/pkl_${MODE}.json"
done

echo
echo "########## REPORT ##########"
python3 src/report.py \
  --results-dir results \
  --prefix pkl_