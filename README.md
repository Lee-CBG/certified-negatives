# Certified negative sampling for TCR epitope binding benchmarks

Code and data for the home-size-matched negative sampling construction and the
per-seed certification protocol.

Negative examples in TCR epitope benchmarks are constructed rather than
measured, and the construction leaves regularities a model can read in place of
binding. This repository contains the construction that closes those
regularities, the protocol that certifies when a construction has done so, and
the diagnostics used to find three successive artifacts in a widely used
benchmark.

## Install

```bash
conda env create -f environment.yml
conda activate tcr-ablation
./setup_env.sh cu124        # or cu121 / cpu; pick from nvidia-smi
export PYTHONPATH=src       # required before every run
```

## Data

```bash
./setup_data.sh             # downloads data/TCREpitopePairs.csv from catELMo
./setup_external.sh         # downloads VDJdb and IMMREP23
```

`data/TCREpitopePairs.csv` is 300,016 pairs over 982 epitopes, exactly 50%
positive, with every epitope individually at 50%.

BLOSUM-encoded datasets (`*.pkl`, about 1.2 GB each) are not in the repository.
They rebuild byte-identically from the CSV; see below.

## Rebuilding the certified datasets

```bash
export PYTHONPATH=src
for s in 0 1 2; do
  python3 src/make_matched_pkl.py --csv data/TCREpitopePairs.csv \
      --seed $s --scale 4 --out data/m4_seed${s}.pkl
  python3 src/make_grouped_pkl.py --pkl data/m4_seed${s}.pkl \
      --out data/m4g_seed${s}.pkl --seed $s
done
```

`--scale` is the bucket-width parameter s and defaults to 1.0, so older calls
reproduce older pickles unchanged.

Check these gates on every build:

| gate | expected |
|---|---|
| `misses` | 0 |
| positive rate | 0.5000 |
| receptor overlap train/test | low thousands (partition), 0 (grouped) |

Use `m4g_*.pkl` for the seen-epitope cell and `m4_*.pkl` for unseen. Both run
with `--split partition`.

## Reproducing the main grid

Three conditions per dataset. `--mode tcr_only` zeroes the epitope input and is
the certification measurement.

```bash
mkdir -p logs results
for s in 0 1 2; do
  for mode in full tcr_only; do
    python3 src/train_pkl.py --pkl data/m4g_seed${s}.pkl --mode $mode \
        --split partition --seed $s \
        --save-preds results/cmp4_s4_${mode}_${s}.npz \
        --out results/cmp4_s4_${mode}_${s}.json
  done
done
python3 src/report.py --results-dir results --prefix cmp4_s4
```

Swap `m4g_` for `m4_` to get the unseen-epitope cells.

## The certification protocol

Applying this to your own construction:

```bash
# 1. receptor-only baseline, per seed
python3 src/train_pkl.py --pkl YOUR.pkl --mode tcr_only --split partition \
    --seed 0 --out results/cert_seed0.json

# 2. simulated null for your test set's actual shape
python3 src/measure_macro_null.py --pkls YOUR.pkl --draws 500 \
    --out results/null.json

# 3. covariate panel: does construction metadata predict the label?
python3 src/probe_shipped.py --csv YOUR.csv --name yourdata \
    --null-draws 500 --out results/covariates.json
```

A construction is certified when the receptor-only macro AUROC falls within one
null standard deviation of 0.500 **on every seed**, not on average. A channel
that is anti-predictive on some epitopes and predictive on others averages to
chance while remaining fully present, so an aggregate check passes constructions
that are still broken.

### Choosing s for a new dataset

s = 4 is the value for this dataset, not a universal constant. The position
channel shrinks as s grows while fallback displacement grows, so the two cancel
at a crossing point that depends on the epitope size distribution. Sweep and
certify:

```bash
python3 src/sweep_granularity.py --csv data/TCREpitopePairs.csv \
    --seeds 0 1 2 --scales 1 2 4 8 --null-draws 200 \
    --out results/granularity.json
```

Select on one set of seeds, confirm on held-out seeds.

## Headline results

catELMo MLP (6,275,073 params), BLOSUM62, 3 seeds, macro AUROC per epitope.
Null 0.500, sd 0.0072 to 0.0080.

| construction | seen full | seen tcr_only | unseen full | unseen tcr_only |
|---|---|---|---|---|
| original | n/a | 0.7462 (no gap) | n/a | n/a |
| swap | 0.7093 ± 0.0079 | 0.4492 ± 0.0082 | 0.5548 ± 0.0318 | 0.4802 ± 0.0138 |
| matched s=1 | 0.6903 ± 0.0077 | 0.4953 ± 0.0146 | not run | not run |
| **matched s=4** | **0.7099 ± 0.0128** | **0.5022 ± 0.0007** | **0.5583 ± 0.0195** | **0.5014 ± 0.0106** |

Epitope contribution rises from +0.0009 on the original data to
+0.2076 ± 0.0124 on seen epitopes under the certified construction.

## Other diagnostics

| script | what it measures |
|---|---|
| `src/interpret_ngram.py` | provenance floor; which n-grams carry it |
| `src/fix_swap_pool.py` | home-size channel before and after matching |
| `src/decompose_fallback.py` | same-bucket versus fallback negatives |
| `src/probe_within_bucket.py` | within-bucket position channel |
| `src/diagnose_split_inversion.py` | receptor-level label inversion by split |
| `src/run_external.py` | full pipeline on VDJdb and IMMREP23 |

## Notes and gotchas

- `build_bucketed` in `src/sweep_granularity.py` is the single builder. It
  returns `(df, stats)`, not a DataFrame.
- The label column is `label` before encoding and `binding` after
  `encode_frame` renames it.
- Pooled AUROC is uninterpretable on grouped splits: grouped assigns whole
  receptors, which breaks per-epitope balance, so pooled reads base-rate
  variation rather than ranking quality. Quote macro.
- `probe_shipped.py` silently drops epitopes below `--min-per-class`. Check `n`
  before reading any macro from it.
- Validation partitions are in donor famine on partition splits, so best-val
  checkpoint selection is not meaningful there.
- `mkdir -p logs` before any backgrounded run that redirects into `logs/`.

## Citation

Paper under review. Citation details to follow.
