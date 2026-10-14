#!/usr/bin/env bash
# Runs the whole experiment: audit, three training conditions, report.
set -euo pipefail

CSV="${CSV:-data/TCREpitopePairs.csv}"
ROWS="${ROWS:-120000}"
EPOCHS="${EPOCHS:-12}"
SEED="${SEED:-0}"

if [ ! -f "$CSV" ]; then
  echo "Missing $CSV -- run ./setup_data.sh first."
  exit 1
fi

mkdir -p results
export PYTHONPATH=src

echo "############ STEP 1/3  DATA AUDIT ############"
python3 src/audit.py --csv "$CSV" --max-rows "$ROWS" --seed "$SEED" \
  --out results/audit.json

for MODE in full tcr_only epi_only; do
  echo
  echo "############ STEP 2/3  TRAIN: $MODE ############"
  python3 src/train.py --csv "$CSV" --mode "$MODE" \
    --max-rows "$ROWS" --epochs "$EPOCHS" --seed "$SEED" \
    --out "results/${MODE}.json"
done

echo
echo "############ STEP 3/3  REPORT ############"
python3 src/report.py --results-dir results