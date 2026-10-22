#!/usr/bin/env bash
export PYTHONPATH=src
J=/tmp/s1u_jobs.txt; : > $J; i=0
for s in 0 1 2; do for m in full tcr_only; do
  g=$((1 + i % 3)); i=$((i+1)); echo "$g $s $m" >> $J
done; done
run_one() {
  read -r g s m <<< "$1"
  tag="s1U_${m}_${s}"
  CUDA_VISIBLE_DEVICES=$g python3 src/train_pkl.py \
      --pkl "data/matched_seed${s}.pkl" --mode "$m" --seed "$s" \
      --out "results/${tag}.json" --save-preds "results/${tag}.npz" \
      > "logs/${tag}.log" 2>&1
  echo "done $tag (gpu $g)"
}
export -f run_one
cat $J | xargs -I{} -P 2 bash -c 'run_one "{}"'
echo "S1 UNSEEN DONE"
