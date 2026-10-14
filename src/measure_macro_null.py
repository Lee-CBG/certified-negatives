"""What is the NULL for per-epitope AUROC on swap-negative data?

Everyone, including us, has been comparing macro AUROC against 0.5. This script
measures what it should be compared against instead, using no trained model at
all.

THE OBSERVATION THAT MOTIVATES IT

Receptor-grouped split, zero receptor overlap, n-gram floor on the CDR3 alone:

    pooled   0.5080 / 0.5051 / 0.5156     ABOVE chance
    macro    0.4733 / 0.4642 / 0.4475     BELOW chance
    bands    1-5: 0.44   ...   101+: 0.49  every band below

Pooled and macro disagreeing in sign is a Simpson-type signature. It points at
how per-epitope averaging weights comparisons, not at a feature that separates
the classes. Length, publicness and donor-pool weight were each tested as
standalone predictors and all three came back at chance (0.4985, 0.5098,
0.5099), with floor-score correlations of +0.039, -0.015 and -0.063.

THE HYPOTHESIS

In the test set, positives and negatives are THE SAME RECEPTORS. A receptor is a
positive for the epitope it binds and a negative for the epitopes it was swapped
into, and the donor pool is dominated by the largest epitopes (the top ten hold
45-70% of it).

So a frequent receptor appears:

    as a POSITIVE inside its own, usually large, epitope -- where its influence
        is diluted across roughly n_pos x n_neg comparisons
    as a NEGATIVE inside several other epitopes, many of them small -- where it
        can be a quarter of the negative side on its own

Macro AUROC weights every epitope equally, so the high-leverage appearances are
disproportionately the negative ones. Any non-constant score function should
then be pushed below 0.5, worst in the smallest epitopes.

If that is the mechanism, it needs NO trained model. Random scores should do it.

THE TEST

    per-receptor random   each DISTINCT receptor draws one score, reused
                          everywhere it appears. This is what any deterministic
                          function of the sequence does, including the floor
                          model and the MLP.
    per-row random        a fresh score for every row, so receptor identity
                          carries nothing.

Per-row is the control and must come out at 0.5. Per-receptor carries the
structure and nothing else -- no sequence features, no training, no labels used
in any way. If it reproduces the floor's ~0.46 and its size banding, the
sub-chance macro is a property of the EVALUATION on this construction, not a
finding about any model.

WHAT EACH OUTCOME MEANS

    per-receptor lands near the floor  -> the null is structural. It can be
        computed for any dataset in seconds, and every macro AUROC on swapped
        negatives must be quoted against it. §4.6 and §9.5 get rewritten in
        those terms, and the correction applies to IMMREP and Liao et al. too.

    per-receptor lands at 0.5          -> structure alone is not enough, the
        floor's deficit is a learned channel after all, and the hunt continues
        with saved predictions from the MLP.

Both splits are simulated, because the null depends on the epitope-size
distribution of the test set and those differ: the partition split holds whole
epitopes, the grouped split samples rows from nearly every epitope.

USAGE

    export PYTHONPATH=src
    python3 src/measure_macro_null.py --out results/macro_null_sim.json
    python3 src/measure_macro_null.py --draws 500        # tighter CI

CPU only, no training. Under a minute per seed at the default 200 draws.
"""

import argparse
import gc
import json
import os
import re
import statistics as st

import numpy as np
import pandas as pd

SIZE_BINS = [0, 5, 10, 25, 100, 10 ** 9]
SIZE_LABELS = ["1-5", "6-10", "11-25", "26-100", "101+"]


# -- fast tie-safe AUROC -------------------------------------------------------

def _avg_ranks(a):
    """Average ranks, ties shared. Equivalent to scipy.stats.rankdata."""
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc(y, s):
    """Mann-Whitney AUROC. y is 0/1."""
    r = _avg_ranks(s)
    npos = float(y.sum())
    nneg = float(len(y) - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


# -- data ----------------------------------------------------------------------

def load_slim(path):
    df = pd.read_pickle(path)
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("epi", "epitope"):
            rename[c] = "epi"
        if lc in ("tcr", "cdr3"):
            rename[c] = "tcr"
        if lc in ("binding", "label"):
            rename[c] = "binding"
        if lc == "partition":
            rename[c] = "partition"
    df = df.rename(columns=rename)
    keep = ["epi", "tcr", "binding"] + (
        ["partition"] if "partition" in df.columns else [])
    out = df[keep].copy()
    del df
    gc.collect()
    return out


def receptor_grouped_split(df, seed, test_frac=0.15, val_frac=0.10):
    rng = np.random.default_rng(seed)
    counts = df["tcr"].value_counts()
    receptors = np.array(sorted(str(r) for r in counts.index), dtype=object)
    sizes = {str(r): int(c) for r, c in counts.items()}
    receptors = rng.permutation(receptors)
    total = len(df)
    targets = {"test": total * test_frac,
               "val": total * val_frac,
               "train": total * (1.0 - test_frac - val_frac)}
    assigned = {k: [] for k in targets}
    filled = {k: 0 for k in targets}
    for r in receptors:
        part = min(targets, key=lambda k: filled[k] / targets[k])
        assigned[part].append(r)
        filled[part] += sizes[r]
    return df[df["tcr"].isin(set(assigned["test"]))].reset_index(drop=True)


# -- simulation ----------------------------------------------------------------

def prepare(te, min_per_class=3):
    """Row indices per qualifying epitope, plus receptor codes for the test set."""
    y = te["binding"].to_numpy().astype(np.int64)
    codes = pd.factorize(te["tcr"])[0]
    groups = []
    for ep, idx in te.groupby("epi").indices.items():
        yl = y[idx]
        p, n = int(yl.sum()), int(len(yl) - yl.sum())
        if p < min_per_class or n < min_per_class:
            continue
        groups.append((idx, yl, p))
    return y, codes, groups


def simulate(te, draws, seed, mode, min_per_class=3):
    """Macro and pooled AUROC under random scores. mode: receptor | row."""
    y, codes, groups = prepare(te, min_per_class)
    if not groups:
        return None
    rng = np.random.default_rng(seed + 9_000)
    n_rec = int(codes.max()) + 1
    npos = np.array([g[2] for g in groups])

    macros, pooleds, band_acc = [], [], {b: [] for b in SIZE_LABELS}
    bands = np.asarray(pd.cut(npos, SIZE_BINS, labels=SIZE_LABELS), dtype=object)

    for _ in range(draws):
        if mode == "receptor":
            s = rng.random(n_rec)[codes]
        elif mode == "row":
            s = rng.random(len(y))
        else:
            raise ValueError(mode)
        a = np.array([auc(yl, s[idx]) for idx, yl, _ in groups])
        macros.append(float(a.mean()))
        pooleds.append(auc(y, s))
        for b in SIZE_LABELS:
            m = bands == b
            if m.any():
                band_acc[b].append(float(a[m].mean()))

    return {
        "mode": mode,
        "draws": draws,
        "n_epitopes": len(groups),
        "macro_mean": round(float(np.mean(macros)), 4),
        "macro_sd": round(float(np.std(macros)), 4),
        "macro_p025": round(float(np.percentile(macros, 2.5)), 4),
        "macro_p975": round(float(np.percentile(macros, 97.5)), 4),
        "frac_draws_below_half": round(float(np.mean(np.array(macros) < 0.5)), 4),
        "pooled_mean": round(float(np.mean(pooleds)), 4),
        "by_size_band": {b: round(float(np.mean(v)), 4)
                         for b, v in band_acc.items() if v},
        "band_n_ep": {b: int((bands == b).sum()) for b in SIZE_LABELS
                      if (bands == b).any()},
    }


def show(tag, r):
    if r is None:
        print(f"    {tag:<26} no qualifying epitopes")
        return
    bands = "".join(f"{r['by_size_band'].get(b, float('nan')):>9.4f}"
                    for b in SIZE_LABELS)
    print(f"    {tag:<26}{r['macro_mean']:>9.4f}{r['macro_sd']:>8.4f}"
          f"  [{r['macro_p025']:.4f}, {r['macro_p975']:.4f}]"
          f"{r['pooled_mean']:>10.4f}   {bands}")


def seed_from_path(path, fallback=0):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkls", nargs="+",
                    default=[f"data/swap_seed{s}.pkl" for s in (0, 1, 2)])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for path in args.pkls:
        if not os.path.exists(path):
            print(f"!! missing {path}, skipping")
            continue
        seed = seed_from_path(path)
        print(f"\n{'=' * 84}\n{path}   (seed {seed}, {args.draws} draws)\n{'=' * 84}")
        df = load_slim(path)
        r = results.setdefault(f"seed{seed}", {"seed": seed})

        print(f"\n  {'':<26}{'macro':>9}{'sd':>8}{'  95% interval':<20}"
              f"{'pooled':>8}   " + "".join(f"{b:>9}" for b in SIZE_LABELS))

        for split_name, te in (
            ("partition", df[df.partition == "test"].reset_index(drop=True)
             if "partition" in df.columns else None),
            ("grouped", receptor_grouped_split(df, seed)),
        ):
            if te is None:
                continue
            n_rec = te["tcr"].nunique()
            print(f"\n  {split_name} test: {len(te):,} rows, "
                  f"{te.epi.nunique()} epitopes, {n_rec:,} distinct receptors "
                  f"({len(te) / n_rec:.2f} rows each)")
            for mode in ("receptor", "row"):
                sim = simulate(te, args.draws, seed, mode, args.min_per_class)
                r[f"{split_name}_{mode}"] = sim
                show(f"random score per {mode}", sim)

        del df
        gc.collect()

    # -- verdict ---------------------------------------------------------------
    print(f"\n{'=' * 84}\nVERDICT\n{'=' * 84}")

    for split_name in ("partition", "grouped"):
        rec = [v[f"{split_name}_receptor"]["macro_mean"] for v in results.values()
               if v.get(f"{split_name}_receptor")]
        row = [v[f"{split_name}_row"]["macro_mean"] for v in results.values()
               if v.get(f"{split_name}_row")]
        if not rec:
            continue
        print(f"\n{split_name}:")
        print(f"  per-row null      {st.mean(row):.4f}   "
              f"(control: must be ~0.5000)")
        print(f"  per-receptor null {st.mean(rec):.4f} +- {st.pstdev(rec):.4f}"
              f"   ({', '.join(f'{x:.4f}' for x in rec)})")

        if abs(st.mean(row) - 0.5) > 0.01:
            print("  !! the per-row control is off chance. Something is wrong with")
            print("     the simulation itself; do not read the per-receptor number.")
        elif st.mean(rec) < 0.49:
            print("  -> STRUCTURAL. Random scores with no training, no features and")
            print("     no labels reproduce a sub-chance macro. This is the null for")
            print("     per-epitope AUROC on this construction. Quote every macro")
            print("     against it, never against 0.5.")
        elif st.mean(rec) > 0.51:
            print("  -> Structure pushes macro ABOVE chance here, which is the")
            print("     opposite of the floor. The floor's deficit is not this.")
        else:
            print("  -> At chance. Structure alone does NOT explain the floor's")
            print("     deficit, so it is a learned channel. Next step is saved")
            print("     predictions from the MLP's tcr_only runs.")

    print("\nCompare against the measured floor macro on the grouped split:")
    print("  0.4733 / 0.4642 / 0.4475,  mean 0.4617")
    print("and the MLP tcr_only macro on partition:")
    print("  0.4902 / 0.4607 / 0.4897,  mean 0.4802")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()