"""Rewrite a swap pkl's partition column to a RECEPTOR-GROUPED split.

WHY THIS EXISTS

The seen-epitope ceiling in §9.5 (macro 0.6878) was measured with
`train_pkl.py --split random`, and that split is not usable on this data.
Partition-local swap pins each receptor's label total near 50/50, and the median
receptor appears about 2.1 times in 300k rows. Split those rows at random and a
receptor's train and test label rates anti-correlate by sampling without
replacement -- if train takes its positive row, test is left holding its
negative. Measured: corr = -0.913, and 96% of receptors that are all-negative in
train are positive in test. The n-gram floor reaches macro 0.32 on it.

Grouping by receptor fixes that. Whole receptors go to one partition, so there
is no per-receptor label total left to split, while epitopes still appear on both
sides -- about 98% of test epitopes are also in train. That is the seen-epitope /
unseen-receptor cell, which is also the axis Liao et al. report models handle
fine, so it tests their claim directly on our data.

WHAT IT PRODUCES

A pkl identical to the input except for the `partition` column, so
`train_pkl.py --split partition` consumes it with no code changes and
`run_swap_experiment.sh`'s job queue works the same way.

    python3 src/make_grouped_pkl.py --pkl data/swap_seed0.pkl \
        --seed 0 --out data/grouped_seed0.pkl

DISK. Each pkl is ~1.24 GB, so three seeds costs about 3.7 GB on top of what is
already there. Check `df -h` first. These are derivable from the swap pkls in
under a minute each, so they are safe to delete once the runs are done.

INTERPRETING THE RESULT

Null macro AUROC for this construction was measured by simulation at 0.500 with
sd 0.007 on grouped-shaped test sets (measure_macro_null.py, 200 draws of random
per-receptor scores). Quote the ceiling against that, and read the tcr_only
condition as the empirical null -- it carries any learned receptor channel that
the random-score simulation does not.
"""

import argparse
import os

import numpy as np
import pandas as pd


def receptor_grouped_assignment(df, seed, test_frac=0.15, val_frac=0.10):
    """Assign whole receptors to partitions by proportional fill.

    Same assignment rule data.py uses for epitopes: each receptor goes to
    whichever partition is currently furthest below its row target. Receptor
    frequencies are far less skewed than epitope frequencies (~2.1 rows each),
    so this is better behaved here than there, but the shared code path keeps
    the two comparable.

    Receptors are materialised as owned Python strings and SORTED before
    permuting, for the two reasons data.py documents: rng.shuffle can fail
    silently on a pandas index array, and value_counts breaks ties by encounter
    order, which would leak row order into the split.
    """
    rng = np.random.default_rng(seed)
    counts = df["tcr"].value_counts()
    receptors = np.array(sorted(str(r) for r in counts.index), dtype=object)
    sizes = {str(r): int(c) for r, c in counts.items()}
    receptors = rng.permutation(receptors)

    total = len(df)
    targets = {
        "test": total * test_frac,
        "val": total * val_frac,
        "train": total * (1.0 - test_frac - val_frac),
    }
    assigned = {k: set() for k in targets}
    filled = {k: 0 for k in targets}

    for r in receptors:
        part = min(targets, key=lambda k: filled[k] / targets[k])
        assigned[part].add(r)
        filled[part] += sizes[r]

    lookup = {}
    for name, members in assigned.items():
        for r in members:
            lookup[r] = name
    return df["tcr"].map(lookup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="an existing swap pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"reading {args.pkl} ...")
    df = pd.read_pickle(args.pkl)

    tcr_col = "tcr" if "tcr" in df.columns else "cdr3"
    epi_col = "epi" if "epi" in df.columns else "epitope"
    lab_col = "binding" if "binding" in df.columns else "label"
    work = df.rename(columns={tcr_col: "tcr", epi_col: "epi", lab_col: "binding"})

    print(f"  {len(df):,} rows, {work.epi.nunique()} epitopes, "
          f"{work.tcr.nunique():,} distinct receptors "
          f"({len(df) / work.tcr.nunique():.2f} rows each)")

    df["partition"] = receptor_grouped_assignment(work, args.seed).to_numpy()
    work["partition"] = df["partition"].to_numpy()

    tr = work[work.partition == "train"]
    va = work[work.partition == "val"]
    te = work[work.partition == "test"]

    # The two properties this construction exists to provide.
    assert not (set(tr.tcr) & set(te.tcr)), "receptor leaked train/test"
    assert not (set(tr.tcr) & set(va.tcr)), "receptor leaked train/val"
    assert not (set(va.tcr) & set(te.tcr)), "receptor leaked val/test"

    shared = len(set(tr.epi) & set(te.epi))
    frac_seen = shared / max(te.epi.nunique(), 1)

    print("\nVERIFY")
    print(f"  receptor overlap train/test: 0 (asserted)")
    print(f"  test epitopes: {te.epi.nunique()}, of which {shared} seen in "
          f"train ({100 * frac_seen:.1f}%)")
    rates = te.groupby("epi")["binding"].mean()
    print(f"  test positive rate {te.binding.mean():.4f}, "
          f"{int((rates.sub(0.5).abs() > 0.05).sum())} epitopes more than "
          f"5pp off 50%")

    if frac_seen < 0.90:
        print("\n  !! fewer than 90% of test epitopes appear in train. This split")
        print("     has drifted toward epitope-disjoint and is NOT the seen-")
        print("     epitope control it is meant to be. Do not run on it.")

    # Rows are already interleaved in the swap pkls, but re-shuffling costs
    # nothing and keeps the float32 tie-breaking guard in place (see §7).
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_pickle(args.out)
    print(f"\nwrote {args.out}")
    for name, g in df.groupby("partition"):
        sub = g.rename(columns={tcr_col: "tcr", epi_col: "epi",
                                lab_col: "binding"})
        print(f"  {name:<6} {len(g):>8,} pairs, "
              f"{sub.epi.nunique():>4} epitopes, "
              f"{100 * sub.binding.mean():.1f}% positive")


if __name__ == "__main__":
    main()