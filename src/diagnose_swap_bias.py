"""Why is tcr_only macro AUROC BELOW chance on the swap data?

Measured symptom, six runs out of six:

    split       seed 0   seed 1   seed 2
    random      0.4783   0.4770   0.4660
    partition   0.4902   0.4607   0.4897

Pooled AUROC is 0.50 by construction -- every repeated receptor is ~50/50, so
the receptor string carries no MARGINAL label information. Macro is a different
question: it asks whether, WITHIN one epitope, receptor features separate that
epitope's own binders from the receptors swapped in as its negatives. Nothing in
the construction forces that to 0.5, and it is coming out consistently under.

THE HYPOTHESIS THIS SCRIPT TESTS

`tcr_swap_negatives` draws the donor pool proportional to positive occurrence:

    pool = pos["cdr3"].to_numpy()

so a receptor's chance of being drawn as somebody's negative is proportional to
how often it appears as a positive, and epitopes with many positives dominate
the pool. The negatives of EVERY epitope are therefore enriched for receptors
that bind the partition's LARGEST epitopes.

A model trained on that learns "looks like a big-epitope binder -> negative".
Now score a large test epitope: its own positives are, by definition, exactly
those receptors. They get pushed toward negative, and per-epitope AUROC lands
below 0.5.

The prediction that distinguishes this from noise is not the mean. It is the
SLOPE: per-epitope AUROC should DECLINE with epitope size, with the largest
epitopes furthest below 0.5 and small epitopes at or above it.

Note that <7% of test receptors appear in train at all (overlap is ~1.4k-1.7k
sequences out of ~27k distinct test receptors), so this cannot be per-receptor
label memorisation. It has to travel through sequence features, which is why the
architecture-free floor model is the right instrument.

WHAT IT RUNS

  [1] pool concentration   -- how top-heavy the donor pool is, per partition
  [2] within-epitope drift -- pool weight of a test epitope's negatives vs its
                              own positives, and how that tracks epitope size
  [3] floor by size band   -- char n-gram LR on the CDR3 string alone, trained
                              on train, per-epitope AUROC on test, banded by
                              epitope size, plus a rank correlation

[3] is the verdict. [1] and [2] are the mechanism behind it.

If the floor reproduces the sub-chance macro AND the slope is negative, this is
a property of the negative construction and `tcr_swap.py` needs fixing -- draw
the donor pool uniformly over DISTINCT receptors instead of over occurrences, or
accept the quota shortfall rather than redrawing.

If the floor sits at or above 0.5, the data is fine and the sub-chance macro
belongs to the MLP. That is the branch where the --save-preds rerun is needed.

USAGE

    export PYTHONPATH=src
    python3 src/diagnose_swap_bias.py                       # 3 seeds, both splits
    python3 src/diagnose_swap_bias.py --splits partition    # half the runtime
    python3 src/diagnose_swap_bias.py --skip-floor          # [1] and [2] only, seconds
    python3 src/diagnose_swap_bias.py --out results/swap_bias.json

No GPU. Roughly two minutes per floor fit, so ~12 minutes for the default.
Loading is the other cost: each pkl is ~1.24 GB and the embedding columns are
dropped immediately, one seed at a time.
"""

import argparse
import gc
import json
import os
import re
import statistics as st

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SIZE_BINS = [0, 5, 10, 25, 100, 10 ** 9]
SIZE_LABELS = ["1-5", "6-10", "11-25", "26-100", "101+"]


# -- helpers -------------------------------------------------------------------

def spearman(a, b):
    """Rank correlation without scipy (not in environment.yml)."""
    a, b = pd.Series(a), pd.Series(b)
    if len(a) < 3:
        return float("nan")
    ra, rb = a.rank().to_numpy(), b.rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def load_slim(path):
    """Read a swap pkl and keep only the four columns this needs."""
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
    missing = {"epi", "tcr", "binding"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    keep = ["epi", "tcr", "binding"] + (
        ["partition"] if "partition" in df.columns else []
    )
    out = df[keep].copy()
    del df
    gc.collect()
    return out


def split_frames(df, split, seed):
    """Reproduce train_pkl.py's two split modes exactly."""
    if split == "partition":
        if "partition" not in df.columns:
            raise SystemExit("no partition column; this is not a swap pkl")
        tr = df[df.partition == "train"].reset_index(drop=True)
        te = df[df.partition == "test"].reset_index(drop=True)
    elif split == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(df))
        n_test = int(0.15 * len(df))
        n_val = int(0.10 * len(df))
        te = df.iloc[perm[:n_test]].reset_index(drop=True)
        tr = df.iloc[perm[n_test + n_val:]].reset_index(drop=True)
    else:
        raise ValueError(split)
    return tr, te


def per_epitope_auroc(te, scores, min_per_class=3):
    """{epitope: (auroc, n_positive)} for epitopes with enough of both classes."""
    y = te["binding"].to_numpy()
    eps = te["epi"].to_numpy()
    out = {}
    for ep in np.unique(eps):
        m = eps == ep
        yl = y[m]
        npos, nneg = int(yl.sum()), int(len(yl) - yl.sum())
        if npos < min_per_class or nneg < min_per_class:
            continue
        out[ep] = (float(roc_auc_score(yl, scores[m])), npos)
    return out


# -- the three checks ----------------------------------------------------------

def check_pool(tr, te):
    """[1] How top-heavy is the donor pool the negatives were drawn from?"""
    rows = {}
    for name, part in (("train", tr), ("test", te)):
        pos = part[part.binding == 1]
        if not len(pos):
            continue
        sizes = pos["epi"].value_counts()
        rows[name] = {
            "n_positive_rows": int(len(pos)),
            "n_epitopes": int(len(sizes)),
            "pool_share_top1": round(float(sizes.iloc[0] / len(pos)), 4),
            "pool_share_top10": round(float(sizes.head(10).sum() / len(pos)), 4),
            "median_epitope_positives": int(sizes.median()),
            "largest_epitope_positives": int(sizes.iloc[0]),
        }
    return rows


def check_drift(te):
    """[2] Are a test epitope's negatives drawn from higher-weight receptors?

    Pool weight w(r) = how often r appears among that partition's POSITIVES.
    That is the quantity `tcr_swap_negatives` draws proportional to, and it is
    fixed before any negative exists, so comparing it across classes is not
    circular the way comparing raw row counts would be.
    """
    pos = te[te.binding == 1]
    w = pos["tcr"].value_counts()
    t = te.copy()
    t["w"] = t["tcr"].map(w).fillna(0.0)

    means = t.groupby(["epi", "binding"])["w"].mean().unstack()
    if 0 not in means.columns or 1 not in means.columns:
        return None
    d = (means[1] - means[0]).dropna()            # positives minus negatives
    sizes = pos["epi"].value_counts().reindex(d.index)

    return {
        "n_epitopes": int(len(d)),
        "median_pos_minus_neg": round(float(d.median()), 3),
        "mean_pos_minus_neg": round(float(d.mean()), 3),
        "frac_epitopes_negatives_heavier": round(float((d < 0).mean()), 4),
        "spearman_with_epitope_size": round(spearman(sizes, d), 4),
    }


def check_floor(tr, te, min_per_class=3):
    """[3] Architecture-free floor: per-epitope AUROC from the CDR3 alone."""
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=3)
    xtr = vec.fit_transform(tr["tcr"])
    xte = vec.transform(te["tcr"])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(xtr, tr["binding"])
    scores = clf.predict_proba(xte)[:, 1]

    pooled = float(roc_auc_score(te["binding"], scores))
    pe = per_epitope_auroc(te, scores, min_per_class)
    if not pe:
        return {"pooled": round(pooled, 4), "macro": float("nan"), "n_ep": 0}

    aucs = np.array([v[0] for v in pe.values()])
    npos = np.array([v[1] for v in pe.values()])

    band = pd.cut(npos, SIZE_BINS, labels=SIZE_LABELS)
    by_band = (
        pd.DataFrame({"auc": aucs, "band": band})
        .groupby("band", observed=True)["auc"]
        .agg(["mean", "size"])
    )

    return {
        "pooled": round(pooled, 4),
        "macro": round(float(aucs.mean()), 4),
        "n_ep": int(len(aucs)),
        "frac_below_half": round(float((aucs < 0.5).mean()), 4),
        "spearman_size_vs_auroc": round(spearman(npos, aucs), 4),
        "by_size_band": {
            str(b): {"macro": round(float(r["mean"]), 4), "n_ep": int(r["size"])}
            for b, r in by_band.iterrows()
        },
    }


# -- main ----------------------------------------------------------------------

def seed_from_path(path, fallback=0):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkls", nargs="+",
                    default=[f"data/swap_seed{s}.pkl" for s in (0, 1, 2)])
    ap.add_argument("--splits", nargs="+", default=["partition", "random"],
                    choices=["partition", "random"])
    ap.add_argument("--min-per-class", type=int, default=3,
                    help="must match train_pkl.py's macro_auroc filter")
    ap.add_argument("--skip-floor", action="store_true",
                    help="run [1] and [2] only; seconds instead of minutes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}

    for path in args.pkls:
        if not os.path.exists(path):
            print(f"!! missing {path}, skipping")
            continue
        seed = seed_from_path(path)
        print(f"\n{'=' * 74}\n{path}   (seed {seed})\n{'=' * 74}")
        df = load_slim(path)
        print(f"  {len(df):,} rows, {df.epi.nunique()} epitopes, "
              f"{df.tcr.nunique():,} distinct receptors")

        for split in args.splits:
            tr, te = split_frames(df, split, seed)
            key = f"seed{seed}_{split}"
            r = results.setdefault(key, {"seed": seed, "split": split})
            shared = len(set(tr.epi) & set(te.epi))
            print(f"\n  --- {split} --- train {len(tr):,} / test {len(te):,}, "
                  f"{shared} epitopes shared")

            print("\n  [1] DONOR POOL CONCENTRATION")
            r["pool"] = check_pool(tr, te)
            for name, v in r["pool"].items():
                print(f"      {name:<6} {v['n_epitopes']:>4} epitopes, "
                      f"median {v['median_epitope_positives']:>5} pos, "
                      f"largest {v['largest_epitope_positives']:>6}, "
                      f"top-10 hold {100 * v['pool_share_top10']:.1f}% of the pool")

            print("\n  [2] WITHIN-EPITOPE POOL-WEIGHT DRIFT (test)")
            d = check_drift(te)
            r["drift"] = d
            if d is None:
                print("      not computable")
            else:
                print(f"      pool weight, positives minus negatives: "
                      f"median {d['median_pos_minus_neg']:+.3f}  "
                      f"mean {d['mean_pos_minus_neg']:+.3f}")
                print(f"      epitopes whose negatives are heavier: "
                      f"{100 * d['frac_epitopes_negatives_heavier']:.1f}%  "
                      f"(n={d['n_epitopes']})")
                print(f"      rank corr with epitope size: "
                      f"{d['spearman_with_epitope_size']:+.3f}")

            if args.skip_floor:
                continue

            print("\n  [3] PROVENANCE FLOOR, PER EPITOPE, BY SIZE")
            f = check_floor(tr, te, args.min_per_class)
            r["floor"] = f
            print(f"      pooled {f['pooled']:.4f}    "
                  f"macro {f['macro']:.4f} over {f['n_ep']} epitopes    "
                  f"{100 * f.get('frac_below_half', 0):.1f}% below 0.5")
            print(f"      {'size band':<12}{'macro':>9}{'n_ep':>7}")
            for b in SIZE_LABELS:
                if b in f.get("by_size_band", {}):
                    v = f["by_size_band"][b]
                    print(f"      {b:<12}{v['macro']:>9.4f}{v['n_ep']:>7}")
            print(f"      rank corr, epitope size vs per-epitope AUROC: "
                  f"{f['spearman_size_vs_auroc']:+.3f}")

        del df
        gc.collect()

    # -- verdict ---------------------------------------------------------------
    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    if args.skip_floor:
        print("Floor skipped, so no verdict. Rerun without --skip-floor.")
    else:
        for split in args.splits:
            rs = [v for v in results.values()
                  if v["split"] == split and "floor" in v]
            if not rs:
                continue
            macros = [v["floor"]["macro"] for v in rs]
            slopes = [v["floor"]["spearman_size_vs_auroc"] for v in rs]
            slopes = [s for s in slopes if s == s]
            print(f"\n{split}:")
            print(f"  floor macro  {st.mean(macros):.4f} "
                  f"+- {st.pstdev(macros):.4f}   "
                  f"({', '.join(f'{m:.4f}' for m in macros)})")
            if slopes:
                print(f"  size slope   {st.mean(slopes):+.3f}   "
                      f"({', '.join(f'{s:+.3f}' for s in slopes)})")

            sub = st.mean(macros) < 0.495
            neg = bool(slopes) and st.mean(slopes) < -0.15
            if sub and neg:
                print("  -> CONFIRMED. The floor reproduces the sub-chance macro and")
                print("     per-epitope AUROC declines with epitope size. This is the")
                print("     donor pool, not the MLP. Fix tcr_swap.py: draw uniformly")
                print("     over distinct receptors, then rebuild the pkls.")
            elif sub:
                print("  -> PARTLY. The floor is sub-chance, so it is in the data, but")
                print("     the size slope is not there. Something other than pool")
                print("     weighting is driving it; look at [2] and at length.")
            else:
                print("  -> NOT THE DATA. The floor is at or above chance, so the")
                print("     sub-chance macro belongs to the model rather than the")
                print("     construction. This is the branch where the --save-preds")
                print("     rerun is needed.")

    print("\nReminder: pooled 0.50 on this data is true BY CONSTRUCTION and is")
    print("not evidence of anything. Macro is the number that can move.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()