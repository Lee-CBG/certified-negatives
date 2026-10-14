#!/usr/bin/env python3
"""
probe_shipped.py -- covariate attribution for a dataset that ships its own
negatives (IMMREP23, or any epi/tcr/binding file we did not construct).

run_external.py reported home_max macro 0.6273 on IMMREP23 over 20 epitopes,
pro-predictive, opposite in sign to the anti-predictive channel measured on
catELMo and VDJdb swap data.  Two things block reading that:

  1. No null.  Twenty epitopes is few; the macro null sd is wide and has to
     be simulated on these exact partitions, not assumed.

  2. Competing covariates.  IMMREP23's own protocol paper documents a leak
     where TCR frequency correlates inversely with label.  A receptor drafted
     as a negative many times has a lower label rate, and 632 receptors are
     easily memorised from character n-grams.  Draw count and home_max may
     be correlated, so home_max cannot be claimed until draw count is ruled
     out -- the same procedure §4.8 used to isolate home_max on catELMo.

Measures, all as per-epitope (macro) AUROC against a calibrated null:
  home_max      size of the largest epitope the receptor is seen binding
  draw_count    times the receptor appears as a negative
  n_rows        times the receptor appears at all
  publicness    number of distinct epitopes the receptor is seen binding
  length        CDR3 length
and home_max stratified within draw_count quartiles, which is the test that
separates the two.

Also recomputes the provenance floor under a receptor-grouped split, so a
receptor cannot appear in both train and test.

Usage:
  export PYTHONPATH=src
  python3 src/probe_shipped.py --csv data/immrep23_canon.csv \
      --name immrep23 --out results/immrep23_covariates.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def macro_auroc(y, scores, epis, min_per_class=3):
    vals = []
    for e in np.unique(epis):
        m = epis == e
        if (y[m] == 1).sum() < min_per_class or (y[m] == 0).sum() < min_per_class:
            continue
        if len(np.unique(scores[m])) < 2:
            vals.append(0.5)
            continue
        vals.append(float(roc_auc_score(y[m], scores[m])))
    return (float(np.mean(vals)) if vals else float("nan"),
            float(np.std(vals)) if vals else float("nan"),
            len(vals))


def simulate_null(df, n_draws=500, min_per_class=3, seed=0):
    """Per-receptor random scores, which is the shape home_max has."""
    y    = df["binding"].to_numpy()
    epis = df["epi"].to_numpy()
    tcrs = df["tcr"].to_numpy()
    receptors = np.unique(tcrs)
    idx = {c: i for i, c in enumerate(receptors)}
    col = np.array([idx[c] for c in tcrs])

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        v = rng.random(len(receptors))[col]
        m, _, _ = macro_auroc(y, v, epis, min_per_class)
        if not np.isnan(m):
            draws.append(m)
    return {"null_mean": float(np.mean(draws)),
            "null_sd":   float(np.std(draws)),
            "n_draws":   len(draws)}


def covariates(df):
    """Per-receptor covariates, mapped back onto rows."""
    epi_sizes = df.groupby("epi").size().to_dict()
    pos = df[df["binding"] == 1]

    home_max = (pos.groupby("tcr")["epi"]
                .apply(lambda ep: max(epi_sizes.get(e, 0) for e in ep)).to_dict())
    publicness = pos.groupby("tcr")["epi"].nunique().to_dict()
    draw_count = df[df["binding"] == 0].groupby("tcr").size().to_dict()
    n_rows     = df.groupby("tcr").size().to_dict()

    out = pd.DataFrame({
        "home_max":   df["tcr"].map(home_max).fillna(0.0),
        "draw_count": df["tcr"].map(draw_count).fillna(0.0),
        "n_rows":     df["tcr"].map(n_rows).fillna(0.0),
        "publicness": df["tcr"].map(publicness).fillna(0.0),
        "length":     df["tcr"].str.len().astype(float),
    })
    return out


def grouped_floor(df, seed=0, n_splits=5, ngram_range=(1, 4)):
    """n-gram floor with whole receptors held out, so none spans the split."""
    X, y = df["tcr"].values, df["binding"].values
    epis, groups = df["epi"].values, df["tcr"].values

    gkf = GroupKFold(n_splits=n_splits)
    S, Y, E = [], [], []
    for tr, te in gkf.split(X, y, groups=groups):
        vec = TfidfVectorizer(analyzer="char", ngram_range=ngram_range,
                              min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform(X[tr]); Xte = vec.transform(X[te])
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(Xtr, y[tr])
        S.append(clf.predict_proba(Xte)[:, 1]); Y.append(y[te]); E.append(epis[te])

    S, Y, E = np.concatenate(S), np.concatenate(Y), np.concatenate(E)
    m, sd, n = macro_auroc(Y, S, E)
    return {"pooled": float(roc_auc_score(Y, S)), "macro": m,
            "macro_sd": sd, "n_epitopes": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--null-draws", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y, epis = df["binding"].to_numpy(), df["epi"].to_numpy()
    print(f"{args.name}: {len(df):,} rows, {int((y==1).sum())} pos / "
          f"{int((y==0).sum())} neg, {df.epi.nunique()} epitopes, "
          f"{df.tcr.nunique()} receptors\n")

    res = {"name": args.name}

    print("── calibrated null " + "─" * 46)
    null = simulate_null(df, args.null_draws, args.min_per_class)
    res["null"] = null
    print(f"  per-receptor random scores: {null['null_mean']:.4f} "
          f"± {null['null_sd']:.4f}  ({null['n_draws']} draws)\n")

    print("── single covariates " + "─" * 44)
    cov = covariates(df)
    res["covariates"] = {}
    print(f"  {'covariate':<12} {'macro':>8} {'sd':>8} {'z':>7} {'n':>5}")
    print(f"  {'-' * 44}")
    for c in cov.columns:
        m, sd, n = macro_auroc(y, cov[c].to_numpy(), epis, args.min_per_class)
        z = (m - null["null_mean"]) / null["null_sd"] if null["null_sd"] else float("nan")
        res["covariates"][c] = {"macro": m, "sd": sd, "n": n, "z": z}
        print(f"  {c:<12} {m:>8.4f} {sd:>8.4f} {z:>+7.1f} {n:>5}")
    print()

    print("── home_max within draw_count quartiles " + "─" * 25)
    print("  (if home_max survives here, it is not draw frequency in disguise)")
    dc = cov["draw_count"]
    try:
        q = pd.qcut(dc, 4, labels=False, duplicates="drop")
    except ValueError:
        q = pd.Series(np.zeros(len(dc)), index=dc.index)
    res["stratified"] = {}
    for lvl in sorted(pd.unique(q.dropna())):
        m_ = (q == lvl).to_numpy()
        if m_.sum() < 20:
            continue
        mm, ss, nn = macro_auroc(y[m_], cov["home_max"].to_numpy()[m_],
                                 epis[m_], args.min_per_class)
        res["stratified"][f"q{int(lvl)}"] = {"macro": mm, "sd": ss, "n": nn,
                                             "rows": int(m_.sum()),
                                             "draw_range": [float(dc[m_].min()),
                                                            float(dc[m_].max())]}
        print(f"    q{int(lvl)}  draws {dc[m_].min():.0f}-{dc[m_].max():.0f}  "
              f"rows {m_.sum():>5}  home_max macro {mm:.4f}  ({nn} epitopes)")
    print()

    print("── correlation between covariates " + "─" * 31)
    corr = cov.corr(method="spearman")
    res["corr"] = corr.to_dict()
    print(corr.round(3).to_string())
    print()

    print("── provenance floor, receptor-grouped split " + "─" * 21)
    runs = [grouped_floor(df, seed=s) for s in args.seeds]
    res["grouped_floor"] = runs
    for s, r in zip(args.seeds, runs):
        print(f"  seed {s}: pooled {r['pooled']:.4f}  macro {r['macro']:.4f} "
              f"({r['n_epitopes']} epitopes)")
    gm = float(np.mean([r["macro"] for r in runs]))
    res["grouped_floor_macro"] = gm
    z = (gm - null["null_mean"]) / null["null_sd"] if null["null_sd"] else float("nan")
    print(f"  mean macro {gm:.4f}   z = {z:+.1f} sd vs null\n")
    print("  Compare with the row-split floor: any drop is receptor identity")
    print("  being memorised rather than sequence features predicting label.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2, default=float)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()