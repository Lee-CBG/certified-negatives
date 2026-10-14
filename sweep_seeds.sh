#!/usr/bin/env bash
# Runs all three conditions across several seeds and reports the spread on the
# epitope contribution. This is the check that turns a single number into a
# claim -- +0.011 is small, and the split is stochastic.
set -euo pipefail

CSV="${CSV:-data/TCREpitopePairs.csv}"
SEEDS="${SEEDS:-0 1 2 3 4}"
ROWS="${ROWS:-120000}"
EPOCHS="${EPOCHS:-12}"
export PYTHONPATH=src

for S in $SEEDS; do
  mkdir -p "results_seed$S"
  for M in full tcr_only epi_only; do
    echo "=== seed $S / $M ==="
    python3 src/train.py --csv "$CSV" --mode "$M" --seed "$S" \
      --max-rows "$ROWS" --epochs "$EPOCHS" \
      --out "results_seed$S/$M.json" | tail -3
  done
done

python3 - "$SEEDS" <<'PY'
import json, sys, statistics
seeds = sys.argv[1].split()
diffs, macro = [], []
print(f"\n{'seed':>6}{'full':>10}{'cdr3_only':>12}{'epi_contrib':>14}")
for s in seeds:
    f = json.load(open(f"results_seed{s}/full.json"))
    t = json.load(open(f"results_seed{s}/tcr_only.json"))
    d = f["test_auroc"] - t["test_auroc"]
    diffs.append(d)
    macro.append(f["macro_auroc_per_epitope"] - t["macro_auroc_per_epitope"])
    print(f"{s:>6}{f['test_auroc']:>10.4f}{t['test_auroc']:>12.4f}{d:>+14.4f}")
print(f"\nepitope contribution: mean {statistics.mean(diffs):+.4f}"
      f"  sd {statistics.pstdev(diffs):.4f}"
      f"  range [{min(diffs):+.4f}, {max(diffs):+.4f}]")
print(f"on macro AUROC:       mean {statistics.mean(macro):+.4f}"
      f"  sd {statistics.pstdev(macro):.4f}")
PY