#!/usr/bin/env python3
"""
sweep_granularity.py -- find the negative-draw construction that minimises the
total home_max channel.

probe_within_bucket.py showed the matched construction's remaining channel is
driven by position inside each log2 bucket: corr(position, per-epitope AUROC)
= +0.78 to +0.83 across 3 seeds on both catELMo and VDJdb, with bottom-tertile
epitopes at ~0.18 and top-tertile at ~0.80.  Cause: within a bucket the donor
pool stays occurrence-weighted, so an epitope low in its bucket draws donors
skewed high and vice versa.

Finer buckets shrink the window but shrink each pool, pushing slots into
fallback -- and fallbacks separate perfectly.  This sweeps that trade-off and
adds a caliper mode with no bucket boundaries at all.

Modes:
  scale=S   bucket = floor(log2(home_max) * S).  S=1 is the current
            construction; S=2 halves the window, S=4 quarters it.
  nearest   no buckets: take the unblocked donor whose home_max is nearest
            the positive's, occurrence-weighted, walking outward from the
            binary-search position.

Reports for each: home_max macro with a calibrated null, the position
correlation under a FIXED log2 stratification (so modes are comparable),
and matching quality as mean |log2 donor - log2 positive|.

Usage:
  export PYTHONPATH=src
  python3 src/sweep_granularity.py --csv data/vdjdb_canon.csv \
      --seeds 0 1 2 --out results/vdjdb_granularity.json
  python3 src/sweep_granularity.py --csv data/TCREpitopePairs.csv \
      --max-rows 300016 --seeds 0 1 2 --out results/catelmo_granularity.json
"""

import argparse
import bisect
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np
import pandas as pd

import fix_swap_pool as fsp


# ── builders ─────────────────────────────────────────────────────────────────

def build_bucketed(df, seed=0, min_distance=3, scale=1.0):
    """
    tcr_swap_negatives_matched with a tunable bucket width.
    scale=1.0 reproduces the current construction exactly.
    """
    rng = np.random.default_rng(seed)
    pos = df[df.label == 1][["epitope", "cdr3"]].copy()
    epitopes = sorted(pos["epitope"].unique())
    near = fsp._too_similar(epitopes, min_distance, verbose=False)
    observed = pos.groupby("cdr3")["epitope"].agg(set).to_dict()

    hm = fsp.home_max_map(pos)
    lg = np.log2(hm.clip(lower=1))
    bucket_of = (lg * scale).astype(int)

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
        want = defaultdict(list)
        for c in g["cdr3"]:
            want[int(bucket_of[c])].append(c)

        for b, members in want.items():
            for src in members:
                c = take(b, blocked, 60)
                was_fb = False
                if c is None:
                    for d in range(1, len(buckets) + 1):
                        for alt in ((b - d, b + d) if rng.random() < 0.5
                                    else (b + d, b - d)):
                            if alt in pools:
                                c = take(alt, blocked, 60)
                                if c is not None:
                                    break
                        if c is not None:
                            break
                    if c is not None:
                        fallbacks += 1
                        was_fb = True
                if c is None:
                    misses += 1
                    continue
                neg_rows.append((ep, c, float(lg[src]), float(lg[c]), was_fb))

    return _assemble(pos, neg_rows, lg, seed, fallbacks, misses)


def build_nearest(df, seed=0, min_distance=3, window=20000):
    """
    Caliper match: for each positive slot, the unblocked donor whose home_max
    is nearest that positive's, with no bucket boundaries.  The donor pool is
    one entry per positive occurrence, so within-pool weighting is unchanged.
    """
    rng = np.random.default_rng(seed)
    pos = df[df.label == 1][["epitope", "cdr3"]].copy()
    epitopes = sorted(pos["epitope"].unique())
    near = fsp._too_similar(epitopes, min_distance, verbose=False)
    observed = pos.groupby("cdr3")["epitope"].agg(set).to_dict()

    hm = fsp.home_max_map(pos)
    lg = np.log2(hm.clip(lower=1))

    occ = pos["cdr3"].to_numpy()
    order = np.argsort([lg[c] for c in occ], kind="stable")
    pool_cdr3 = occ[order]
    pool_lg = np.array([lg[c] for c in pool_cdr3], dtype=float)
    consumed = np.zeros(len(pool_cdr3), dtype=bool)
    pool_list = pool_lg.tolist()

    neg_rows, misses, far = [], 0, 0
    # interleave epitopes so draw order does not favour whoever goes first
    groups = list(pos.groupby("epitope"))
    rng.shuffle(groups)

    for ep, g in groups:
        blocked = near[ep] | {ep}
        for src in g["cdr3"]:
            target = float(lg[src])
            i = bisect.bisect_left(pool_list, target)
            pick = None
            for d in range(window):
                for j in (i - d, i + d) if d else (i,):
                    if 0 <= j < len(pool_cdr3) and not consumed[j]:
                        c = pool_cdr3[j]
                        if not (observed.get(c, set()) & blocked):
                            pick = (j, c)
                            break
                if pick:
                    break
            if pick is None:
                misses += 1
                continue
            j, c = pick
            consumed[j] = True
            dist = abs(float(pool_lg[j]) - target)
            if dist > 0.25:
                far += 1
            neg_rows.append((ep, c, target, float(pool_lg[j]), dist > 0.25))

    return _assemble(pos, neg_rows, lg, seed, far, misses)


def _assemble(pos, neg_rows, lg, seed, fallbacks, misses):
    neg = pd.DataFrame(neg_rows, columns=["epitope", "cdr3",
                                          "want_lg", "src_lg", "is_fallback"])
    neg["label"] = 0
    p = pos.assign(label=1)
    p["want_lg"] = p["cdr3"].map(lg).astype(float)
    p["src_lg"] = p["want_lg"]
    p["is_fallback"] = False
    out = pd.concat([p, neg], ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    return out, {"fallbacks": int(fallbacks), "misses": int(misses),
                 "mean_abs_lg_dist": float((neg["src_lg"] - neg["want_lg"])
                                           .abs().mean()) if len(neg) else 0.0}


# ── measurement ──────────────────────────────────────────────────────────────

def evaluate(built, min_per_class=3, n_draws=200, seed=0):
    pos = built[built.label == 1][["epitope", "cdr3"]]
    hm = fsp.home_max_map(pos)
    v = built["cdr3"].map(hm).fillna(0.0).to_numpy(dtype=float)
    y = built["label"].to_numpy()
    ep = built["epitope"].to_numpy()
    sizes = pos.groupby("epitope").size().to_dict()

    rows, masks = [], []
    for e in np.unique(ep):
        m = ep == e
        if (y[m] == 1).sum() < min_per_class or (y[m] == 0).sum() < min_per_class:
            continue
        s = sizes.get(e, 1)
        l = float(np.log2(max(s, 1)))
        rows.append({"auroc": fsp.auc(y[m], v[m]), "position": l - int(l)})
        masks.append(m)

    if not rows:
        return {"macro": float("nan"), "z": float("nan"), "n": 0,
                "corr_position": float("nan")}

    tbl = pd.DataFrame(rows)
    macro = float(tbl["auroc"].mean())

    cdr3 = built["cdr3"].to_numpy()
    receptors = np.unique(cdr3)
    idx = {c: i for i, c in enumerate(receptors)}
    col = np.array([idx[c] for c in cdr3])
    rng = np.random.default_rng(seed)
    draws = [float(np.mean([fsp.auc(y[m], rng.random(len(receptors))[col][m])
                            for m in masks])) for _ in range(n_draws)]
    nm, ns = float(np.mean(draws)), float(np.std(draws))

    r = (float(np.corrcoef(tbl["position"], tbl["auroc"])[0, 1])
         if len(tbl) >= 10 else float("nan"))

    return {"macro": macro, "null_mean": nm, "null_sd": ns,
            "z": (macro - nm) / ns if ns else float("nan"),
            "n": len(tbl), "corr_position": r,
            "auroc_sd": float(tbl["auroc"].std())}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-rows", type=int, default=300016)
    ap.add_argument("--min-distance", type=int, default=3)
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--null-draws", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = fsp.subsample(fsp.load_pairs(args.csv), args.max_rows, seed=0)
    print(f"loaded {len(df):,} pairs\n")

    modes = [(f"scale={s:g}", lambda p, sd, s=s:
              build_bucketed(p, seed=sd, min_distance=args.min_distance, scale=s))
             for s in args.scales]
    modes.append(("nearest", lambda p, sd:
                  build_nearest(p, seed=sd, min_distance=args.min_distance)))

    results = defaultdict(dict)
    for seed in args.seeds:
        print(f"{'=' * 78}\nSEED {seed}\n{'=' * 78}")
        tr, va, te = fsp.epitope_disjoint_split(df, seed=seed)
        print(f"  test: {len(te):,} rows, {te.epitope.nunique()} epitopes\n")
        print(f"  {'mode':<12} {'macro':>8} {'z':>7} {'corr':>7} "
              f"{'fb%':>7} {'|dist|':>7} {'sd':>7} {'n':>5}")
        print(f"  {'-' * 68}")
        for name, fn in modes:
            built, counts = fn(te, seed)
            ev = evaluate(built, args.min_per_class, args.null_draws, seed)
            n_neg = int((built.label == 0).sum())
            fb = counts["fallbacks"] / max(n_neg, 1)
            results[name][seed] = {**ev, **counts, "fallback_frac": fb,
                                   "positive_rate": float(built.label.mean())}
            print(f"  {name:<12} {ev['macro']:>8.4f} {ev['z']:>+7.1f} "
                  f"{ev['corr_position']:>+7.3f} {fb:>6.1%} "
                  f"{counts['mean_abs_lg_dist']:>7.3f} "
                  f"{ev['auroc_sd']:>7.3f} {ev['n']:>5}")
        print()

    print(f"{'=' * 78}\nVERDICT\n{'=' * 78}\n")
    print(f"  {'mode':<12} {'macro':>16} {'|z|':>7} {'corr':>8} "
          f"{'fb%':>7} {'auroc sd':>9}")
    print(f"  {'-' * 68}")
    best, best_score = None, 1e9
    for name, _ in modes:
        d = [results[name][s] for s in args.seeds]
        m = st.mean([x["macro"] for x in d])
        sd = st.pstdev([x["macro"] for x in d]) if len(d) > 1 else 0.0
        z = st.mean([abs(x["z"]) for x in d if not np.isnan(x["z"])])
        c = st.mean([x["corr_position"] for x in d
                     if not np.isnan(x["corr_position"])])
        fb = st.mean([x["fallback_frac"] for x in d])
        asd = st.mean([x["auroc_sd"] for x in d])
        print(f"  {name:<12} {m:>9.4f} ± {sd:.4f} {z:>7.1f} {c:>+8.3f} "
              f"{fb:>6.1%} {asd:>9.3f}")
        score = abs(m - 0.5) + 0.5 * abs(c) + 0.3 * asd
        if score < best_score:
            best, best_score = name, score

    print(f"\n  Lowest combined channel: {best}")
    print("  (|macro-0.5| + 0.5*|corr| + 0.3*per-epitope sd;"
          " check the columns yourself before trusting the ranking)")
    print("\n  A mode is only usable if positive rate stayed 0.5000 and misses"
          " are ~0.")
    for name, _ in modes:
        d = [results[name][s] for s in args.seeds]
        print(f"    {name:<12} pos rate {st.mean([x['positive_rate'] for x in d]):.4f}"
              f"   misses {sum(x['misses'] for x in d):,}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({k: {str(s): v for s, v in d.items()}
                       for k, d in results.items()}, fh, indent=2)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()