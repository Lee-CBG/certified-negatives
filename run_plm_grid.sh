#!/usr/bin/env bash
export PYTHONPATH=src
JOBS=/tmp/plm_jobs.txt
: > $JOBS
i=0
for p in esm650 tcrbert; do
  for c in m4g grouped; do
    for s in 0 1 2; do
      for m in full tcr_only; do
        if [ "$c" = "m4g" ] && [ "$s" = "0" ]; then continue; fi
        g=$((1 + i % 3)); i=$((i+1))
        echo "$g $p $c $s $m" >> $JOBS
      done
    done
  done
done
echo "queued $(wc -l < $JOBS) runs"
run_one() {
  read -r g p c s m <<< "$1"
  tag="${p}_${c}_${m}_${s}"
  CUDA_VISIBLE_DEVICES=$g python3 src/train_pkl.py \
      --pkl "data/${p}_${c}_seed${s}.pkl" --mode "$m" --seed "$s" \
      --out "results/${tag}.json" --save-preds "results/${tag}.npz" \
      > "logs/${tag}.log" 2>&1
  echo "done $tag (gpu $g)"
}
export -f run_one
cat $JOBS | xargs -I{} -P 6 bash -c 'run_one "{}"'
echo "ALL DONE"
