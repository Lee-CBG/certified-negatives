"""What makes per-epitope AUROC sit below chance when only the receptor is shown?

THE EFFECT, as established by diagnose_split_inversion.py

Receptor-grouped split (epitopes shared, zero receptor overlap), n-gram floor
on the CDR3 string alone:

    pooled  0.5080 / 0.5051 / 0.5156        at chance
    macro   0.4733 / 0.4642 / 0.4475        below chance, 57-68% of epitopes
    by size 1-5: 0.44 ... 101+: 0.49        worst in SMALL epitopes

No receptor crosses the split, so this is not memorisation and not the
hypergeometric inversion that ruins the plain random split. It is a sequence
feature that separates a test epitope's own binders from the receptors swapped
in as its negatives, in a direction that is consistent across epitopes.

The MLP's tcr_only macro (0.4902 / 0.4607 / 0.4897 partition, 0.4783 / 0.4770 /
0.4660 random) has the same signature, so it is reading the same thing.

WHY THE DIRECTION IS CONSISTENT, IN PRINCIPLE

Within one epitope the two classes come from different places:

    positives = that epitope's own binders, a narrow epitope-specific cluster
    negatives = drawn from the partition-wide donor pool, i.e. the AVERAGE
                receptor, since `pool = pos["cdr3"].to_numpy()` draws in
                proportion to positive occurrence

So any receptor property on which epitope-specific binders differ SYSTEMATICALLY
from the repertoire average will separate the classes the same way in every
epitope, and will not cancel when averaged. Small epitopes should show it most,
because their binders are the most private -- which matches the size banding.

THE CANDIDATES, AND THE FIX EACH ONE IMPLIES

    length       CDR3 length. Fix: draw length-matched negatives per epitope.
    publicness   how many distinct epitopes the receptor is observed binding.
                 Fix: draw the pool uniformly over DISTINCT receptors.
    weight       positive-row count, i.e. the donor-pool draw weight itself.
                 Same fix as publicness.

Each is scored here as a standalone predictor: per-epitope AUROC using that
covariate alone as the score. A covariate that lands near 0.46 with the same
size banding is the channel. One near 0.50 is not involved.

Pooled AUROC is reported for each too. The signature to look for is pooled at
chance with macro below it -- that combination is what says the covariate is
balanced overall but misaligned WITHIN epitopes, which is the whole effect.

WHAT THIS DECIDES

    a covariate reproduces it  -> fixable bug in tcr_swap.py; match on that
                                  covariate when drawing negatives, rebuild,
                                  re-measure. The null returns to 0.5.
    nothing reproduces it      -> the channel is distributed across sequence
                                  features and cannot be matched away. Then the
                                  null is irreducible and must be REPORTED
                                  rather than removed: quote every macro against
                                  the measured tcr_only baseline, never 0.5.

Either way the grouped split stays the right seen-epitope control. It just needs
its null measured beside it, which is what the tcr_only condition does.

USAGE

    export PYTHONPATH=src
    python3 src/diagnose_macro_null.py --out results/macro_null.json
    python3 src/diagnose_macro_null.py --no-floor       # skip the n-gram refit

CPU only. About a minute per seed without --no-floor's two-minute refit.
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
    out = df[["epi", "tcr", "binding"]].copy()
    del df
    gc.collect()
    return out


def receptor_grouped_split(df, seed, test_frac=0.15, val_frac=0.10):
    """Whole receptors to partitions; epitopes stay shared. See
    diagnose_split_inversion.py for why this is the right ceiling control."""
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

    tr = df[df["tcr"].isin(set(assigned["train"]))].reset_index(drop=True)
    te = df[df["tcr"].isin(set(assigned["test"]))].reset_index(drop=True)
    assert not (set(tr["tcr"]) & set(te["tcr"]))
    return tr, te


def score_report(te, scores, min_per_class=3):
    """Pooled AUROC, macro AUROC, and macro by epitope size, for one score."""
    y = te["binding"].to_numpy()
    eps = te["epi"].to_numpy()
    pooled = float(roc_auc_score(y, scores))

    aucs, npos = [], []
    for ep in np.unique(eps):
        m = eps == ep
        yl = y[m]
        p, n = int(yl.sum()), int(len(yl) - yl.sum())
        if p < min_per_class or n < min_per_class:
            continue
        aucs.append(float(roc_auc_score(yl, scores[m])))
        npos.append(p)
    if not aucs:
        return {"pooled": round(pooled, 4), "macro": float("nan"), "n_ep": 0}

    aucs, npos = np.array(aucs), np.array(npos)
    band = pd.cut(npos, SIZE_BINS, labels=SIZE_LABELS)
    by = (pd.DataFrame({"auc": aucs, "band": band})
          .groupby("band", observed=True)["auc"].agg(["mean", "size"]))
    return {
        "pooled": round(pooled, 4),
        "macro": round(float(aucs.mean()), 4),
        "n_ep": int(len(aucs)),
        "frac_below_half": round(float((aucs < 0.5).mean()), 4),
        "by_size_band": {str(b): round(float(r["mean"]), 4)
                         for b, r in by.iterrows()},
    }


def class_gap(te, col):
    """Within-epitope mean of `col`, positives minus negatives."""
    m = te.groupby(["epi", "binding"])[col].mean().unstack()
    if 0 not in m.columns or 1 not in m.columns:
        return None
    d = (m[1] - m[0]).dropna()
    return {"median": round(float(d.median()), 4),
            "mean": round(float(d.mean()), 4),
            "frac_neg_higher": round(float((d < 0).mean()), 4),
            "n_epitopes": int(len(d))}


def seed_from_path(path, fallback=0):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkls", nargs="+",
                    default=[f"data/swap_seed{s}.pkl" for s in (0, 1, 2)])
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--no-floor", action="store_true",
                    help="skip the n-gram refit and its correlations")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    covs = ["length", "publicness", "weight"]

    for path in args.pkls:
        if not os.path.exists(path):
            print(f"!! missing {path}, skipping")
            continue
        seed = seed_from_path(path)
        print(f"\n{'=' * 76}\n{path}   (seed {seed})\n{'=' * 76}")
        df = load_slim(path)

        # Covariates come from the FULL dataset, so they do not depend on the
        # split. Both are computed from POSITIVE rows only: negative rows exist
        # because of the draw, so counting them would be circular.
        pos_all = df[df.binding == 1]
        publicness = pos_all.groupby("tcr")["epi"].nunique()
        weight = pos_all["tcr"].value_counts()

        tr, te = receptor_grouped_split(df, seed)
        te = te.copy()
        te["length"] = te["tcr"].str.len().astype(float)
        te["publicness"] = te["tcr"].map(publicness).fillna(0.0)
        te["weight"] = te["tcr"].map(weight).fillna(0.0)
        print(f"  train {len(tr):,} / test {len(te):,}, "
              f"{len(set(tr.epi) & set(te.epi))} epitopes shared, "
              f"receptor overlap 0")

        r = results.setdefault(f"seed{seed}", {"seed": seed})

        print(f"\n  SINGLE-COVARIATE PER-EPITOPE AUROC")
        print(f"    {'covariate':<12}{'pooled':>9}{'macro':>9}{'n_ep':>7}"
              f"{'<0.5':>8}   " + "".join(f"{b:>9}" for b in SIZE_LABELS))
        for c in covs:
            rep = score_report(te, te[c].to_numpy(), args.min_per_class)
            r[c] = rep
            bands = "".join(
                f"{rep.get('by_size_band', {}).get(b, float('nan')):>9.4f}"
                for b in SIZE_LABELS)
            print(f"    {c:<12}{rep['pooled']:>9.4f}{rep['macro']:>9.4f}"
                  f"{rep['n_ep']:>7}{100 * rep.get('frac_below_half', 0):>7.1f}%"
                  f"   {bands}")

        print(f"\n  WITHIN-EPITOPE CLASS GAP (positives minus negatives)")
        for c in covs:
            g = class_gap(te, c)
            r[f"{c}_gap"] = g
            if g:
                print(f"    {c:<12} median {g['median']:>+9.4f}   "
                      f"mean {g['mean']:>+9.4f}   "
                      f"negatives higher in "
                      f"{100 * g['frac_neg_higher']:.1f}% of epitopes")

        if not args.no_floor:
            vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=3)
            xtr = vec.fit_transform(tr["tcr"])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(xtr, tr["binding"])
            s = clf.predict_proba(vec.transform(te["tcr"]))[:, 1]
            rep = score_report(te, s, args.min_per_class)
            r["floor"] = rep
            bands = "".join(
                f"{rep.get('by_size_band', {}).get(b, float('nan')):>9.4f}"
                for b in SIZE_LABELS)
            print(f"\n  FLOOR MODEL, for comparison")
            print(f"    {'floor':<12}{rep['pooled']:>9.4f}{rep['macro']:>9.4f}"
                  f"{rep['n_ep']:>7}{100 * rep.get('frac_below_half', 0):>7.1f}%"
                  f"   {bands}")
            print(f"    correlation of floor score with each covariate:")
            for c in covs:
                v = te[c].to_numpy()
                cc = (float(np.corrcoef(s, v)[0, 1])
                      if v.std() > 0 else float("nan"))
                r[f"floor_corr_{c}"] = round(cc, 4)
                print(f"      {c:<12}{cc:>+9.3f}")

        del df, tr, te, pos_all
        gc.collect()

    # -- verdict ---------------------------------------------------------------
    print(f"\n{'=' * 76}\nVERDICT\n{'=' * 76}")
    target = None
    if all("floor" in v for v in results.values()):
        fm = [v["floor"]["macro"] for v in results.values()]
        target = st.mean(fm)
        print(f"\nfloor macro to explain: {target:.4f} "
              f"({', '.join(f'{x:.4f}' for x in fm)})")

    print(f"\n{'covariate':<14}{'macro mean':>12}{'sd':>9}   verdict")
    print("-" * 76)
    best, best_gap = None, 1.0
    for c in covs:
        ms = [v[c]["macro"] for v in results.values() if c in v]
        if not ms:
            continue
        m, sd = st.mean(ms), st.pstdev(ms)
        if m < 0.485:
            note = "BELOW CHANCE -- candidate channel"
            if target is not None and abs(m - target) < best_gap:
                best, best_gap = c, abs(m - target)
        elif m > 0.515:
            note = "above chance -- inverted, still a channel"
        else:
            note = "at chance -- not involved"
        print(f"{c:<14}{m:>12.4f}{sd:>9.4f}   {note}")

    print()
    if best is not None:
        print(f"Closest single covariate to the floor: {best}.")
        if best in ("publicness", "weight"):
            print("Both point at the same line in tcr_swap.py:")
            print("    pool = pos[\"cdr3\"].to_numpy()")
            print("which draws negatives in proportion to positive occurrence,")
            print("so every epitope's negatives are enriched for the repertoire's")
            print("most frequent receptors while its positives are its own.")
            print("Fix: draw uniformly over DISTINCT receptors, rebuild the pkls,")
            print("then re-run this script -- the macro null should return to 0.5.")
        else:
            print("Fix: draw length-matched negatives within each epitope, rebuild,")
            print("then re-run this script and check the null returns to 0.5.")
    else:
        print("No single covariate reproduces the floor's macro. The channel is")
        print("spread across sequence features and cannot be matched away one at")
        print("a time. Then the null is a property of swapped negatives and must")
        print("be REPORTED, not removed:")
        print()
        print("  - quote every macro against the measured tcr_only baseline")
        print("  - never against 0.5")
        print("  - §4.6 and §9.5 both need rewriting on that basis")
        print()
        print("That is a publishable result in its own right: per-epitope metrics")
        print("on swap-negative benchmarks have a null below chance, so reported")
        print("per-epitope AUROCs across the field are understated against the")
        print("wrong baseline.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()