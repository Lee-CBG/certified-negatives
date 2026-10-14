#!/usr/bin/env python3
"""
decompose_fallback.py -- split the matched construction's residual home_max
channel into same-bucket and fallback negatives.

fix_swap_pool.py reports a single matched home_max macro (0.5624 on VDJdb).
This asks whether that residual is carried entirely by the bucket fallbacks,
which is what the construction predicts: a large epitope exhausts its own
bucket (its receptors are blocked from their own epitope), falls back to a
lower bucket, and the resulting negative has lower home_max than the positives
it is matched against.

Rebuilds the matched negatives with identical RNG consumption -- the tagging
is pure bookkeeping, no extra rng calls -- so the numbers reproduce
fix_swap_pool.py exactly.

Usage:
  export PYTHONPATH=src
  python3 src/decompose_fallback.py --csv data/vdjdb_canon.csv \
      --seeds 0 1 2 --out results/vdjdb_fallback_decomp.json

  # also dump the matched partitions as CSV for downstream use
  python3 src/decompose_fallback.py --csv data/vdjdb_canon.csv \
      --seeds 0 1 2 --write-csv data/vdjdb_matched_seed{seed}_{part}.csv
"""

import argparse
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np
import pandas as pd

import fix_swap_pool as fsp


# ── builder: verbatim copy with fallback tagging ─────────────────────────────

def build_matched_tagged(df, seed=0, min_distance=3, verbose=False):
    """
    tcr_swap_negatives_matched with two extra output columns.

    Returns a frame with [epitope, cdr3, label, src_bucket, is_fallback]
    where for negatives src_bucket is the bucket actually drawn from and
    is_fallback marks slots that could not be filled from the wanted bucket.
    Positives carry their own bucket and is_fallback = False.

    RNG call sequence is identical to the original, so the built negatives
    are the same rows in the same order.
    """
    rng = np.random.default_rng(seed)
    pos = df[df.label == 1][["epitope", "cdr3"]].copy()
    epitopes = sorted(pos["epitope"].unique())
    near = fsp._too_similar(epitopes, min_distance, verbose=False)
    observed = pos.groupby("cdr3")["epitope"].agg(set).to_dict()

    hm = fsp.home_max_map(pos)
    bucket_of = np.log2(hm.clip(lower=1)).astype(int)

    pools = defaultdict(list)
    for cdr3 in pos["cdr3"].to_numpy():
        pools[int(bucket_of[cdr3])].append(cdr3)
    buckets = sorted(pools)
    for b in buckets:
        pools[b] = rng.permutation(np.array(pools[b], dtype=object))
    cursor = {b: 0 for b in buckets}

    def take(b, blocked, tries):
        p = pools[b]
        if len(p) == 0:
            return None
        for _ in range(tries):
            if cursor[b] >= len(p):
                pools[b] = rng.permutation(p)
                p = pools[b]
                cursor[b] = 0
            c = p[cursor[b]]
            cursor[b] += 1
            if not (observed.get(c, set()) & blocked):
                return c
        return None

    neg_rows, misses, fallbacks = [], 0, 0
    for ep, g in pos.groupby("epitope"):
        blocked = near[ep] | {ep}
        want = defaultdict(int)
        for c in g["cdr3"]:
            want[int(bucket_of[c])] += 1

        for b, k in want.items():
            for _ in range(k):
                c = take(b, blocked, 60)
                drawn_from = b                      # bookkeeping only
                was_fallback = False                # bookkeeping only
                if c is None:
                    for d in range(1, len(buckets) + 1):
                        for alt in ((b - d, b + d)
                                    if rng.random() < 0.5
                                    else (b + d, b - d)):
                            if alt in pools:
                                c = take(alt, blocked, 60)
                                if c is not None:
                                    drawn_from = alt     # bookkeeping only
                                    break
                        if c is not None:
                            break
                    if c is not None:
                        fallbacks += 1
                        was_fallback = True
                if c is None:
                    misses += 1
                    continue
                neg_rows.append((ep, c, b, drawn_from, was_fallback))

    neg = pd.DataFrame(neg_rows, columns=["epitope", "cdr3",
                                          "want_bucket", "src_bucket",
                                          "is_fallback"])
    neg["label"] = 0

    pos_t = pos.assign(label=1)
    pos_t["want_bucket"] = pos_t["cdr3"].map(bucket_of).astype(int)
    pos_t["src_bucket"]  = pos_t["want_bucket"]
    pos_t["is_fallback"] = False

    out = pd.concat([pos_t, neg], ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    if verbose:
        print(f"    {len(out):,} pairs, {100 * out.label.mean():.2f}% positive, "
              f"{fallbacks:,} bucket fallbacks, {misses:,} unplaced")
    return out, {"fallbacks": int(fallbacks), "misses": int(misses)}


# ── measurement ──────────────────────────────────────────────────────────────

def home_max_macro(built, min_per_class=3):
    """
    Per-epitope AUROC of home_max as a score, matching fix_swap_pool.measure().
    home_max is computed from the built partition's own positives.
    """
    pos = built[built.label == 1][["epitope", "cdr3"]]
    hm  = fsp.home_max_map(pos)
    v   = built["cdr3"].map(hm).fillna(0.0).to_numpy(dtype=float)
    y   = built["label"].to_numpy()
    ep  = built["epitope"].to_numpy()

    vals, gaps = [], []
    for e in np.unique(ep):
        m = ep == e
        ypos, yneg = (y[m] == 1), (y[m] == 0)
        if ypos.sum() < min_per_class or yneg.sum() < min_per_class:
            continue
        vals.append(fsp.auc(y[m], v[m]))
        gaps.append(float(v[m][ypos].mean() - v[m][yneg].mean()))

    return {
        "macro":      float(np.mean(vals)) if vals else float("nan"),
        "sd":         float(np.std(vals))  if vals else float("nan"),
        "n_epitopes": len(vals),
        "median_gap": float(np.median(gaps)) if gaps else None,
    }


def decompose(built, min_per_class=3):
    """home_max macro on all negatives, same-bucket only, fallback only."""
    pos = built[built.label == 1]
    neg = built[built.label == 0]

    same = pd.concat([pos, neg[~neg.is_fallback]], ignore_index=True)
    fb   = pd.concat([pos, neg[ neg.is_fallback]], ignore_index=True)

    n_fb = int(neg.is_fallback.sum())
    direction = neg.loc[neg.is_fallback, "src_bucket"] - \
                neg.loc[neg.is_fallback, "want_bucket"]

    return {
        "n_neg":            int(len(neg)),
        "n_fallback":       n_fb,
        "fallback_frac":    round(n_fb / max(len(neg), 1), 4),
        "fallback_down":    int((direction < 0).sum()),
        "fallback_up":      int((direction > 0).sum()),
        "fallback_mean_shift": float(direction.mean()) if n_fb else None,
        "all":        home_max_macro(built, min_per_class),
        "same_bucket": home_max_macro(same, min_per_class),
        "fallback":    home_max_macro(fb,   min_per_class),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-rows", type=int, default=300016)
    ap.add_argument("--min-distance", type=int, default=3)
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--parts", nargs="+", default=["test"],
                    choices=["train", "test"],
                    help="which partitions to decompose (default: test)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-csv", default=None,
                    help="template, e.g. data/vdjdb_matched_seed{seed}_{part}.csv")
    args = ap.parse_args()

    df = fsp.subsample(fsp.load_pairs(args.csv), args.max_rows, seed=0)
    print(f"loaded {len(df):,} pairs\n")

    results = {}
    for seed in args.seeds:
        print(f"{'=' * 72}\nSEED {seed}\n{'=' * 72}")
        tr, va, te = fsp.epitope_disjoint_split(df, seed=seed)
        parts = {"train": tr, "test": te}
        r = results.setdefault(f"seed{seed}", {"seed": seed})

        for name in args.parts:
            part = parts[name]
            print(f"\n  [{name}] {len(part):,} rows, "
                  f"{part.epitope.nunique()} epitopes")
            built, counts = build_matched_tagged(
                part, seed=seed, min_distance=args.min_distance, verbose=True)
            d = decompose(built, args.min_per_class)
            d["build_counts"] = counts
            r[name] = d

            print(f"\n    fallbacks {d['n_fallback']:,} / {d['n_neg']:,} "
                  f"({d['fallback_frac']:.1%})   "
                  f"down {d['fallback_down']:,}  up {d['fallback_up']:,}  "
                  f"mean shift {d['fallback_mean_shift']:+.2f} buckets")
            for k in ("all", "same_bucket", "fallback"):
                s = d[k]
                print(f"    {k:<12} macro {s['macro']:.4f} ± {s['sd']:.4f}"
                      f"   gap {s['median_gap']:>10.1f}"
                      f"   ({s['n_epitopes']} epitopes)")

            if args.write_csv:
                path = args.write_csv.format(seed=seed, part=name)
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                built.rename(columns={"epitope": "epi", "cdr3": "tcr",
                                      "label": "binding"})[
                    ["epi", "tcr", "binding"]].to_csv(path, index=False)
                print(f"    → {path}")
        print()

    # ── verdict ──────────────────────────────────────────────────────────────
    print(f"{'=' * 72}\nVERDICT\n{'=' * 72}\n")
    part = args.parts[-1]
    for k in ("all", "same_bucket", "fallback"):
        vals = [results[f"seed{s}"][part][k]["macro"] for s in args.seeds]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            print(f"  {k:<12} {st.mean(vals):.4f} ± "
                  f"{st.pstdev(vals) if len(vals) > 1 else 0:.4f}   "
                  f"({', '.join(f'{v:.4f}' for v in vals)})")

    sb = [results[f"seed{s}"][part]["same_bucket"]["macro"] for s in args.seeds]
    sb = [v for v in sb if not np.isnan(v)]
    fb = [results[f"seed{s}"][part]["fallback"]["macro"] for s in args.seeds]
    fb = [v for v in fb if not np.isnan(v)]

    print()
    if sb and abs(st.mean(sb) - 0.5) < 0.03:
        print("Same-bucket negatives are at chance. The residual is carried")
        print("entirely by bucket fallbacks, which are forced by epitopes whose")
        print("own bucket holds too few unblocked donors. Irreducible at this")
        print("dataset scale -- report the fallback fraction alongside the")
        print("matched baseline rather than treating the construction as broken.")
    elif sb:
        print(f"Same-bucket negatives are at {st.mean(sb):.4f}, not chance.")
        print("Fallbacks are not the whole story -- log2 bucketing is too coarse,")
        print("or home_max proxies something that survives the match. Try finer")
        print("buckets before reporting.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()