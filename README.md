# Home-size-matched negative sampling for TCR epitope binding benchmarks

Code and data for the home-size-matched negative sampling construction and the
per-seed screening protocol.

Negative examples in TCR epitope benchmarks are constructed rather than
measured, and the construction leaves regularities a model can read in place of
binding. This repository contains the construction, the protocol that screens
whether a given construction has closed them, and the diagnostics used to find
three successive artifacts in a widely used benchmark.

**A screening verdict is a joint property of a construction, an architecture and
a split design.** No construction here passes on both split designs, and the
construction that passes under a BLOSUM62 encoding does not pass under either
frozen protein language model we tested. We release datasets annotated by
screening outcome, not a certified dataset.

> **Note on the repository name.** The URL still says `certified-negatives`.
> The word "certified" has been retired from this work: the protocol screens, it
> does not certify. Rename pending.

## Install

```
conda env create -f environment.yml
conda activate tcr-ablation
./setup_env.sh cu124        # or cu121 / cpu; pick from nvidia-smi
export PYTHONPATH=src       # required before every run
```

## Data

```
./setup_data.sh             # downloads data/TCREpitopePairs.csv from catELMo
./setup_external.sh         # downloads VDJdb and IMMREP23
```

`data/TCREpitopePairs.csv` is 300,016 pairs over 982 epitopes, exactly 50%
positive, with every epitope individually at 50%.

BLOSUM-encoded datasets (`*.pkl`, about 1.2 GB each) are not in the repository.
They rebuild byte-identically from the CSV; see below.

## Rebuilding the matched datasets

```
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

| gate                        | expected                               |
| --------------------------- | -------------------------------------- |
| `misses`                    | 0                                      |
| positive rate               | 0.5000                                 |
| receptor overlap train/test | low thousands (partition), 0 (grouped) |

Use `m4g_*.pkl` for the seen-epitope cell and `m4_*.pkl` for unseen. Both run
with `--split partition`.

## Reproducing the main grid

Three conditions per dataset. `--mode tcr_only` zeroes the epitope input and is
the screening measurement.

```
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

## The screening protocol

```
# 1. receptor-only baseline, per seed
python3 src/train_pkl.py --pkl YOUR.pkl --mode tcr_only --split partition \
    --seed 0 --out results/cert_seed0.json

# 2. simulated null for YOUR test set's actual shape -- not a borrowed range
python3 src/measure_macro_null.py --pkls YOUR.pkl --draws 500 \
    --out results/null.json

# 3. covariate panel: does construction metadata predict the label?
python3 src/probe_shipped.py --csv YOUR.csv --name yourdata \
    --null-draws 500 --out results/covariates.json
```

A construction **passes the screen** when the receptor-only macro AUROC falls
within one null standard deviation of 0.500 **on every seed**, not on average.
This is a heuristic operational rule, not a calibrated test of equivalence or of
artifact absence. Its false-flag rate on artifact-free constructions, its joint
behaviour across trained runs and the independence of its outcomes are all
unestablished, and the random-score reference is a different variance component
from across-run model variability.
A channel that is anti-predictive on some epitopes and predictive on others
averages to chance while remaining fully present, so an aggregate check passes
constructions that are still broken.

**Step 3 is not optional.** A pass states that the macro stays inside the band
on this test set, which per-epitope effects can achieve by cancelling. It does
not state that the model ignores the sampler: on the primary dataset at s = 4
the receptor-only model correlates with draw count at Spearman -0.219 to -0.316
across seeds (all p < 1e-3) while the macro sits at chance. Run the covariate
panel beside the screen.

### The null must be simulated per test set

Null widths measured here, by test-set family. A single range borrowed from one
family and applied to another changes verdicts.

| family                               | null sd          |
| ------------------------------------ | ---------------- |
| matched grouped (`m4g_*`, `mg_*`)    | 0.0048 to 0.0052 |
| swap grouped (`grouped_*`)           | 0.0071 to 0.0079 |
| background grouped (`baseg_*`)       | 0.0076 to 0.0080 |
| matched partition (`m4_*`, `matched_*`) | 0.0058 to 0.0116 |
| background partition (`base_*`)      | 0.0119 to 0.0160 |

Earlier versions of this README quoted "sd 0.0072 to 0.0080" for the whole
table. That was the swap-grouped number applied everywhere, and under it
Matched s=1 seen (0.4953) passes when against its own null it does not.

### Choosing s for a new dataset

s = 4 is the value for this dataset, not a universal constant. The position
channel shrinks as s grows while fallback displacement grows, so the two cancel
at a crossing point that depends on the epitope size distribution **and on the
split design**: s = 4 passes on seen epitopes and s = 1 on unseen. Sweep and
screen:

```
python3 src/sweep_granularity.py --csv data/TCREpitopePairs.csv \
    --seeds 0 1 2 --scales 1 2 4 8 --null-draws 200 \
    --out results/granularity.json
```

Select on one set of seeds, confirm on held-out seeds. The sweep scores each
construction with the home-size metadata probe, so it uses no trained model and
no encoding.

## Headline results

catELMo MLP (6,275,073 params), BLOSUM62, 3 seeds, macro AUROC per epitope,
sample standard deviation. **Each cell is scored against its own simulated
null** (table above).

| construction    | seen full           | seen tcr\_only      | unseen full         | unseen tcr\_only    | passes |
| --------------- | ------------------- | ------------------- | ------------------- | ------------------- | ------ |
| original        | 0.7879 ± 0.0048     | 0.7647 ± 0.0019     | 0.7489 ± 0.0144     | 0.7586 ± 0.0146     | neither |
| swap            | 0.7093 ± 0.0097     | 0.4492 ± 0.0100     | 0.5548 ± 0.0318     | 0.4802 ± 0.0138     | seen: no; unseen: unassessed |
| matched s=1     | 0.6903 ± 0.0077     | 0.4953 ± 0.0146     | 0.5535 ± 0.0291     | **0.5005 ± 0.0046** | unseen |
| **matched s=4** | **0.7099 ± 0.0128** | **0.5022 ± 0.0007** | **0.5583 ± 0.0195** | 0.5014 ± 0.0106     | seen |

The ablation contrast (full minus tcr_only macro AUROC, paired per seed) rises
from **+0.0232 ± 0.0066** on the original data to **+0.2076 ± 0.0124** under the
matched draw at s = 4, same split design, same encoder.

Earlier versions of this README quoted +0.0009 for the original construction.
That figure was pooled, single-seed and from the unseen split, so it was not
comparable to the per-epitope seen-split number it was being compared against.
The pooled three-seed difference is +0.0131 ± 0.0038. The contrast is a descriptive
difference between two models, not a decomposition into biological and
artifactual components.

`swap unseen` has no per-seed verdict: `assemble_grid.py` has no mapping for
that cell. It is marked unassessed rather than failing.

### Three representations, seen split

Frozen mean-pooled embeddings, identical head. The shared tail after the input
layers is identically 4,206,593 parameters in all three.

| representation | construction | full | tcr\_only | ablation contrast | passes |
| -------------- | ------------ | ---- | --------- | ------------ | ------ |
| BLOSUM62   | swap        | 0.7093 ± 0.0097 | 0.4492 ± 0.0100 | +0.2601 ± 0.0198 | no |
| ESM-2 650M | swap        | 0.6894 ± 0.0050 | 0.4729 ± 0.0123 | +0.2164 ± 0.0131 | no |
| TCR-BERT   | swap        | 0.7003 ± 0.0103 | 0.4679 ± 0.0200 | +0.2324 ± 0.0232 | no |
| BLOSUM62   | matched s=4 | 0.7099 ± 0.0128 | **0.5022 ± 0.0007** | +0.2076 ± 0.0124 | **yes** |
| ESM-2 650M | matched s=4 | 0.6726 ± 0.0097 | 0.5055 ± 0.0031 | +0.1671 ± 0.0067 | no |
| TCR-BERT   | matched s=4 | 0.7027 ± 0.0065 | 0.5047 ± 0.0020 | +0.1980 ± 0.0054 | no |

The audit half of this work replicates across all three representations. The
remedy half does not.

## Other diagnostics

| script                            | what it measures                            |
| --------------------------------- | ------------------------------------------- |
| `src/audit.py`                    | provenance floor (**pooled** AUROC), class overlap, surface stats |
| `src/interpret_ngram.py`          | which n-grams carry the floor               |
| `src/fix_swap_pool.py`            | home-size channel before and after matching |
| `src/decompose_fallback.py`       | same-bucket versus fallback negatives       |
| `src/probe_within_bucket.py`      | within-bucket position channel              |
| `src/diagnose_split_inversion.py` | receptor-level label inversion by split     |
| `src/assemble_grid.py`            | full grid, per-test-set nulls, per-seed verdicts |
| `src/receptor_channel.py`         | receptor-level score-covariate correlations |
| `src/isolation_test.py`           | within-epitope sequence isolation gap       |
| `src/run_external.py`             | full pipeline on VDJdb and IMMREP23         |

## Notes and gotchas

- **The provenance floor from `audit.py` is pooled AUROC, not macro.**
  `check_provenance_leakage` calls `roc_auc_score` once over the whole test set.
  The 0.7326 figure is therefore comparable to the pooled full-model 0.7701 on
  the same splits, and must not be read as a per-epitope macro.
  `run_external.py` reports both pooled and macro; on IMMREP23 they agree to
  0.0004 because every epitope there carries the same 1:5 class ratio.
- `build_bucketed` in `src/sweep_granularity.py` is the single builder. It
  returns `(df, stats)`, not a DataFrame.
- The label column is `label` before encoding and `binding` after
  `encode_frame` renames it.
- Pooled AUROC is uninterpretable on grouped splits: grouped assigns whole
  receptors, which breaks per-epitope balance (453 of 667 grouped test epitopes
  end more than 5 percentage points off 50%), so pooled reads base-rate
  variation rather than ranking quality. Quote macro.
- `macro_auroc` applies `min_per_class=3`. This gates every macro and is why the
  number of epitopes entering the macro varies across seeds and constructions.
  `probe_shipped.py` silently drops epitopes below `--min-per-class`; check `n`
  before reading any macro from it.
- The n-gram floor is blind by construction on data with balanced per-receptor
  labels: the gradients cancel, the model emits a constant and macro reads
  0.5000. That is no measurement, not no signal. Check `max|coef|`.
- Validation partitions are in donor famine on partition splits, so best-val
  checkpoint selection is not meaningful there. **See the training protocol note
  below.**
- `mkdir -p logs` before any backgrounded run that redirects into `logs/`.
- `--max-rows 300016`, not 300000. The file has 300,016 rows and anything lower
  silently subsamples, which changes epitope counts, which changes the split.

## Known gaps

These are open and are stated in the paper as limitations:

- **Training protocol is not documented here.** Optimiser, learning rate, batch
  size, epoch count and the checkpoint-selection rule used on partition splits
  need to be written down, since best-val selection is not meaningful on those
  splits. Reviewers will ask.
- The screen's false-positive rate and its power against a planted channel of
  known magnitude have not been characterised. Three seeds is not enough to
  state either.
- PLM unseen cells not run; layers 1 and 3 not run under the PLMs.
- About 12% of negative rows are duplicate (epitope, receptor) pairs from donor
  reuse. Unmeasured as a channel.
- VDJdb and IMMREP23 have not been rebuilt at s = 4.

## Housekeeping

Three files at the top level are committed shell-redirection accidents and
should be deleted: `ort -u`, `ses in staged files ==="`, and
`ecret-ish patterns in staged files ==="`. They look like fragments of a broken
pre-commit secret scan; check whether that scan ever ran.

## Citation

Paper under review. Citation details to follow.