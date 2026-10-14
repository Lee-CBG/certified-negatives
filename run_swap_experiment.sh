#!/usr/bin/env bash
# Full experiment: 3 seeds x 2 split modes x 3 ablation conditions on the
# TCR-swap data, queued across a set of GPUs.
#
#   ./run_swap_experiment.sh                 # GPUs 1 2 3, seeds 0 1 2
#   GPUS="1 2 3" SEEDS="0 1 2" ./run_swap_experiment.sh
#   DRY=1 ./run_swap_experiment.sh           # print the job list and exit
#
# Split modes:
#   partition  epitope-disjoint, negatives built inside each partition
#              -> performance on epitopes never seen in training
#   random     rows split at random, epitopes appear in train AND test
#              -> performance on epitopes seen in training
#
# The gap between them is the memorisation result. Running the ablation under
# both localises it: if the epitope contributes on seen epitopes and not on
# unseen ones, the failure is specifically epitope generalisation.
#
# Already-finished runs are skipped, so this is safe to re-run after a crash.
set -euo pipefail

CSV="${CSV:-data/TCREpitopePairs.csv}"
GPUS="${GPUS:-1 2 3}"
SEEDS="${SEEDS:-0 1 2}"
DRY="${DRY:-0}"
export PYTHONPATH=src

mkdir -p results data

# ---------- build one swap pkl per seed -------------------------------------
for S in $SEEDS; do
  PKL="data/swap_seed${S}.pkl"
  if [ -f "$PKL" ]; then
    echo "have $PKL"
  elif [ "$DRY" = "1" ]; then
    echo "WOULD BUILD $PKL"
  else
    echo "building $PKL ..."
    python3 src/make_swap_pkl.py --csv "$CSV" --mode swap \
        --seed "$S" --out "$PKL"
  fi
done
echo

# ---------- assemble the job list -------------------------------------------
JOBS=()
for S in $SEEDS; do
  for SPLIT in partition random; do
    for MODE in full tcr_only epi_only; do
      TAG="swap_${SPLIT}_s${S}_${MODE}"
      [ -f "results/${TAG}.json" ] && continue      # already done
      JOBS+=("${S}:${SPLIT}:${MODE}:${TAG}")
    done
  done
done

echo "${#JOBS[@]} runs to do"
if [ "$DRY" = "1" ]; then
  printf '  %s\n' "${JOBS[@]}"
  exit 0
fi
[ "${#JOBS[@]}" -eq 0 ] && { echo "nothing to do"; exit 0; }
echo

# ---------- queue across GPUs -----------------------------------------------
read -r -a GPU_ARR <<< "$GPUS"
declare -A SLOT_PID SLOT_TAG

launch() {
  local gpu="$1" spec="$2"
  IFS=':' read -r S SPLIT MODE TAG <<< "$spec"
  echo "  GPU $gpu  <-  $TAG"
  CUDA_VISIBLE_DEVICES="$gpu" python3 src/train_pkl.py \
      --pkl "data/swap_seed${S}.pkl" \
      --mode "$MODE" --split "$SPLIT" --seed "$S" --tf32 \
      --out "results/${TAG}.json" \
      > "results/log_${TAG}.txt" 2>&1 &
  SLOT_PID[$gpu]=$!
  SLOT_TAG[$gpu]="$TAG"
}

J=0
for g in "${GPU_ARR[@]}"; do
  [ "$J" -lt "${#JOBS[@]}" ] || break
  launch "$g" "${JOBS[$J]}"
  J=$((J+1))
done

FAILED=()
while [ "$J" -lt "${#JOBS[@]}" ] || [ "${#SLOT_PID[@]}" -gt 0 ]; do
  sleep 10
  for g in "${!SLOT_PID[@]}"; do
    pid="${SLOT_PID[$g]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      tag="${SLOT_TAG[$g]}"
      if wait "$pid" 2>/dev/null; then
        echo "  [ok]     $tag"
      else
        echo "  [FAILED] $tag  -> results/log_${tag}.txt"
        FAILED+=("$tag")
      fi
      unset 'SLOT_PID[$g]' 'SLOT_TAG[$g]'
      if [ "$J" -lt "${#JOBS[@]}" ]; then
        launch "$g" "${JOBS[$J]}"
        J=$((J+1))
      fi
    fi
  done
done

echo
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "${#FAILED[@]} run(s) failed:"
  printf '  %s\n' "${FAILED[@]}"
  echo "Re-run this script to retry only those."
  exit 1
fi

# ---------- summarise --------------------------------------------------------
echo "########## SUMMARY ##########"
python3 - "$SEEDS" <<'PY'
import json, os, sys, statistics as st

seeds = sys.argv[1].split()
def load(split, s, mode):
    p = f"results/swap_{split}_s{s}_{mode}.json"
    return json.load(open(p)) if os.path.exists(p) else None

for split, label in [("random", "epitopes SEEN in training"),
                     ("partition", "epitopes UNSEEN")]:
    print(f"\n{split}  ({label})")
    print(f"{'seed':>6}{'full':>10}{'tcr_only':>11}{'epi_only':>11}"
          f"{'contrib':>10}{'macro_full':>12}")
    rows = []
    for s in seeds:
        f, t, e = (load(split, s, m) for m in ("full", "tcr_only", "epi_only"))
        if not (f and t and e):
            print(f"{s:>6}  incomplete")
            continue
        c = f["test_auroc"] - t["test_auroc"]
        rows.append((f["test_auroc"], t["test_auroc"], e["test_auroc"], c,
                     f["macro_auroc_per_epitope"]))
        print(f"{s:>6}{f['test_auroc']:>10.4f}{t['test_auroc']:>11.4f}"
              f"{e['test_auroc']:>11.4f}{c:>+10.4f}"
              f"{f['macro_auroc_per_epitope']:>12.4f}")
    if len(rows) > 1:
        cols = list(zip(*rows))
        names = ["full", "tcr_only", "epi_only", "contrib", "macro_full"]
        print("  mean " + "  ".join(
            f"{n}={st.mean(c):.4f}+-{st.pstdev(c):.4f}"
            for n, c in zip(names, cols)))

# The headline comparison
rs = [load("random", s, "full") for s in seeds]
ps = [load("partition", s, "full") for s in seeds]
if all(rs) and all(ps):
    gp = [r["test_auroc"] - p["test_auroc"] for r, p in zip(rs, ps)]
    gm = [r["macro_auroc_per_epitope"] - p["macro_auroc_per_epitope"]
          for r, p in zip(rs, ps)]
    print(f"\nSEEN minus UNSEEN (the memorisation gap)")
    print(f"  pooled AUROC  {st.mean(gp):+.4f} +- {st.pstdev(gp):.4f}")
    print(f"  macro  AUROC  {st.mean(gm):+.4f} +- {st.pstdev(gm):.4f}")
    print("\n  Macro is the one to quote: every epitope is exactly 50% positive,")
    print("  so epitope identity carries no label information and within-epitope")
    print("  ranking cannot be inflated by memorising which epitope it is.")
PY