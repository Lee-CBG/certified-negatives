#!/usr/bin/env python3
"""
probe_within_bucket.py -- calibrate the same-bucket residual and test whether
within-bucket position explains it.

decompose_fallback.py showed the matched construction's residual splits into
bucket fallbacks (perfect or near-perfect separation, 1-6 epitopes) and a
same-bucket remainder displaced from chance by ~0.025 -- below on catELMo
(0.4748), above on VDJdb (0.5208).

Two questions:

  1. Is that displacement real?  Calibrated against a per-receptor random-score
     null on the same partitions, not against 0.5 by eye.

  2. Does within-bucket position explain it?  log2 buckets span a factor of
     two and the within-bucket pool stays occurrence-weighted, so an epitope
     low in its bucket draws donors skewed high (AUROC below 0.5) and an
     epitope at the top of its bucket draws donors from below (above 0.5).
     Predicts a positive correlation between position and per-epitope AUROC.

Usage:
  export PYTHONPATH=src
  python3 src/probe_within_bucket.py --csv data/vdjdb_canon.csv \
      --seeds 0 1 2 --out results/vdjdb_within_bucket.json
  python3 src/probe_within_bucket.py --csv data/TCREpitopePairs.csv \
      --max-rows 300016 --seeds 0 1 2 --out results/catelmo_within_bucket.json
"""

import argparse
import json
import os
import statistics as st

import numpy as np
import pandas as pd

import fix_swap_pool as fsp
from decompose_fallback import build_matched_tagged


def per_epitope(built, min_per_class=3):
    """
    Per-epitope home_max AUROC on same-bucket negatives only, with each
    epitope's size, bucket and position inside that bucket.
    """
    pos = built[built.label == 1][["epitope", "cdr3"]]
    hm  = fsp.home_max_map(pos)

    neg  = built[built.label == 0]
    same = pd.concat([built[built.label == 1], neg[~neg.is_fallback]],
                     ignore_index=True)

    v  = same["cdr3"].map(hm).fillna(0.0).to_numpy(dtype=float)
    y  = same["label"].to_numpy()
    ep = same["epitope"].to_numpy()

    sizes = pos.groupby("epitope").size().to_dict()

    rows = []
    for e in np.unique(ep):
        m = ep == e
        ypos, yneg = (y[m] == 1), (y[m] == 0)
        if ypos.sum() < min_per_class or yneg.sum() < min_per_class:
            continue
        size = sizes.get(e, 0)
        if size < 1:
            continue
        lg  = float(np.log2(size))
        rows.append({
            "epitope":  e,
            "size":     int(size),
            "bucket":   int(lg),
            "position": lg - int(lg),          # 0 = bottom of bucket, ~1 = top
            "auroc":    fsp.auc(y[m], v[m]),
            "n_pos":    int(ypos.sum()),
            "n_neg":    int(yneg.sum()),
        })
    return pd.DataFrame(rows)


def macro_null(built, n_draws=200, min_per_class=3, seed=0):
    """
    Null for per-epitope macro AUROC on the same-bucket subset: assign each
    receptor a random score (constant across its rows, as home_max is), then
    recompute macro.  Returns mean and sd over draws.
    """
    neg  = built[built.label == 0]
    same = pd.concat([built[built.label == 1], neg[~neg.is_fallback]],
                     ignore_index=True)

    y    = same["label"].to_numpy()
    ep   = same["epitope"].to_numpy()
    cdr3 = same["cdr3"].to_numpy()

    receptors = np.unique(cdr3)
    idx = {c: i for i, c in enumerate(receptors)}
    col = np.array([idx[c] for c in cdr3])

    keep = []
    for e in np.unique(ep):
        m = ep == e
        if (y[m] == 1).sum() >= min_per_class and (y[m] == 0).sum() >= min_per_class:
            keep.append(e)
    keep = set(keep)
    masks = [(ep == e) for e in keep]

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        rs = rng.random(len(receptors))
        v  = rs[col]
        vals = [fsp.auc(y[m], v[m]) for m in masks]
        if vals:
            draws.append(float(np.mean(vals)))

    return {
        "null_mean": float(np.mean(draws)) if draws else float("nan"),
        "null_sd":   float(np.std(draws))  if draws else float("nan"),
        "n_draws":   len(draws),
        "n_epitopes": len(masks),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-rows", type=int, default=300016)
    ap.add_argument("--min-distance", type=int, default=3)
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--null-draws", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = fsp.subsample(fsp.load_pairs(args.csv), args.max_rows, seed=0)
    print(f"loaded {len(df):,} pairs\n")

    results, all_tables = {}, []
    for seed in args.seeds:
        print(f"{'=' * 72}\nSEED {seed}\n{'=' * 72}")
        tr, va, te = fsp.epitope_disjoint_split(df, seed=seed)
        built, _ = build_matched_tagged(te, seed=seed,
                                        min_distance=args.min_distance,
                                        verbose=False)

        tbl = per_epitope(built, args.min_per_class)
        tbl["seed"] = seed
        all_tables.append(tbl)

        macro = float(tbl["auroc"].mean())
        null  = macro_null(built, args.null_draws, args.min_per_class, seed)
        z     = (macro - null["null_mean"]) / null["null_sd"] if null["null_sd"] else float("nan")

        # correlation of within-bucket position with per-epitope AUROC
        if len(tbl) >= 10:
            r_pos  = float(np.corrcoef(tbl["position"], tbl["auroc"])[0, 1])
            r_size = float(np.corrcoef(np.log2(tbl["size"]), tbl["auroc"])[0, 1])
        else:
            r_pos = r_size = float("nan")

        results[f"seed{seed}"] = {
            "seed": seed,
            "same_bucket_macro": macro,
            "null_mean": null["null_mean"],
            "null_sd":   null["null_sd"],
            "z":         z,
            "n_epitopes": len(tbl),
            "corr_position_auroc": r_pos,
            "corr_logsize_auroc":  r_size,
        }

        print(f"  same_bucket macro {macro:.4f}   "
              f"null {null['null_mean']:.4f} ± {null['null_sd']:.4f}   "
              f"z = {z:+.1f} sd   ({len(tbl)} epitopes)")
        print(f"  corr(within-bucket position, AUROC) = {r_pos:+.3f}")
        print(f"  corr(log2 size, AUROC)              = {r_size:+.3f}")

        # AUROC by position tertile
        if len(tbl) >= 12:
            q = pd.qcut(tbl["position"], 3,
                        labels=["bottom", "middle", "top"], duplicates="drop")
            print("  AUROC by within-bucket position:")
            for lab, g in tbl.groupby(q, observed=True):
                print(f"    {str(lab):<7} n={len(g):>4}  "
                      f"mean position {g['position'].mean():.2f}  "
                      f"AUROC {g['auroc'].mean():.4f}")
        print()

    print(f"{'=' * 72}\nVERDICT\n{'=' * 72}\n")
    macros = [results[f"seed{s}"]["same_bucket_macro"] for s in args.seeds]
    zs     = [results[f"seed{s}"]["z"] for s in args.seeds]
    rs     = [results[f"seed{s}"]["corr_position_auroc"] for s in args.seeds
              if not np.isnan(results[f"seed{s}"]["corr_position_auroc"])]

    print(f"  same_bucket macro  {st.mean(macros):.4f} ± "
          f"{st.pstdev(macros) if len(macros) > 1 else 0:.4f}")
    print(f"  z vs null          {', '.join(f'{z:+.1f}' for z in zs)} sd")
    if rs:
        print(f"  corr(position, AUROC)  {st.mean(rs):+.3f}   "
              f"({', '.join(f'{r:+.3f}' for r in rs)})")
    print()

    if all(abs(z) < 2 for z in zs if not np.isnan(z)):
        print("Same-bucket negatives are at chance within the calibrated null.")
        print("The matched construction's only residual is the bucket fallbacks.")
    elif rs and st.mean(rs) > 0.15:
        print("Same-bucket is displaced from chance AND within-bucket position")
        print("predicts the per-epitope direction. The occurrence-weighted draw")
        print("inside each bucket is the cause: finer buckets (or a within-bucket")
        print("size match) should close it. Worth one more construction pass.")
    else:
        print("Same-bucket is displaced from chance but within-bucket position")
        print("does not explain it. home_max proxies something else that")
        print("survives the match. Report the offset as a measured baseline")
        print("and use the contribution metric (full minus tcr_only), which")
        print("subtracts it out.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        full = pd.concat(all_tables, ignore_index=True)
        results["_per_epitope"] = full.to_dict(orient="records")
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()