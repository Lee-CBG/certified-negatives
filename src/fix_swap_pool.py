"""Close the home-epitope-size channel in the swap-negative construction.

THE DIAGNOSIS (diagnose_home_size.py, 3 seeds, grouped split)

    home_max per-epitope AUROC   0.1674 / 0.1710 / 0.1806     mean 0.1730
    home_max POOLED AUROC        0.5397 / 0.5366 / 0.5360
    within-epitope gap           -3795 (1-5) rising to -991 (101+)
    corr(model score, home_max)  +0.121 / +0.118 / +0.140

`home_max` is the size of the largest epitope a receptor binds. Pooled it says
"bigger home epitope, more likely positive". Within an epitope it says the exact
opposite, overwhelmingly. A model trained with pooled BCE learns the marginal
direction and applies it inside epitopes where the truth is reversed, which is
why a receptor-only baseline lands at ~0.45 macro instead of 0.50.

WHERE EACH DIRECTION COMES FROM

Marginal (+0.54): the blocking rule bars a receptor from being a negative for its
own epitope and its near neighbours. A receptor homed on a 12,940-pair epitope is
barred from the single largest sink of negative quota, so it is drawn as a
negative less often than its pool weight implies and ends up over-represented
among positives.

Conditional (0.17): epitope e's positives have home_max ~ size(e), because most
receptors bind exactly one epitope. Its negatives are drawn proportional to
occurrence, so they come from the LARGEST epitopes no matter how big e is. Small
e gives near-total separation; large e gives almost none. That is the banding.

THE FIX

Draw each epitope's negatives so their home_max distribution MATCHES that of its
own positives, instead of matching the global occurrence-weighted pool. Receptors
are bucketed by log2(home_max); for each epitope the bucket histogram of its
positives becomes the quota for its negatives, with a nearest-bucket fallback
when a bucket cannot fill under the blocking constraints.

This removes a study-size effect, not biology. home_max reflects how many TCRs
happened to be sequenced for an epitope, which is a property of the experiments
in VDJdb and IEDB rather than of receptor chemistry -- a cleaner intervention
than the V/J rung in §4.4, which arguably removed real repertoire structure.

WHAT THIS SCRIPT DOES

Rebuilds partition-local negatives both ways on the same positives and same
partitions, then measures the channel directly:

    home_max per-epitope AUROC      should move 0.17 -> ~0.50
    within-epitope home_max gap     should move -3800 -> ~0
    class balance and blocking      must survive the change

No GPU, no embeddings, no training. If the channel closes, rebuild the pkls with
make_swap_pkl.py using the matched draw and re-run; if it does not, the effect is
not the pool and the measured-baseline route stands.

USAGE

    export PYTHONPATH=src
    python3 src/fix_swap_pool.py --csv data/TCREpitopePairs.csv --seeds 0 1 2
    python3 src/fix_swap_pool.py --csv data/TCREpitopePairs.csv --seeds 0 \
        --out results/fix_pool.json
"""

import argparse
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np
import pandas as pd

from data import epitope_disjoint_split, load_pairs, subsample
from tcr_swap import _too_similar, tcr_swap_negatives


def _avg_ranks(a):
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    r = np.empty(len(a), dtype=np.float64)
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def auc(y, s):
    r = _avg_ranks(s)
    npos = float(y.sum())
    nneg = float(len(y) - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def home_max_map(pos):
    """Receptor -> size of the largest epitope it is observed binding."""
    sizes = pos["epitope"].value_counts()
    tmp = pos.assign(_sz=pos["epitope"].map(sizes))
    return tmp.groupby("cdr3")["_sz"].max()


def tcr_swap_negatives_matched(df, seed=0, min_distance=3, n_buckets=None,
                               verbose=True):
    """Swap negatives whose home_max distribution matches each epitope's own.

    Same contract as tcr_swap_negatives: one negative per positive, no receptor
    paired with an epitope it binds or with a near neighbour of one. The only
    change is WHICH receptor is offered for each slot.
    """
    rng = np.random.default_rng(seed)
    pos = df[df.label == 1][["epitope", "cdr3"]].copy()
    epitopes = sorted(pos["epitope"].unique())
    near = _too_similar(epitopes, min_distance, verbose=False)
    observed = pos.groupby("cdr3")["epitope"].agg(set).to_dict()

    hm = home_max_map(pos)
    bucket_of = np.log2(hm.clip(lower=1)).astype(int)

    # One pool per bucket, each entry a positive OCCURRENCE so that within a
    # bucket the draw stays occurrence-weighted -- only the between-bucket mix
    # is changed.
    pools = defaultdict(list)
    for cdr3 in pos["cdr3"].to_numpy():
        pools[int(bucket_of[cdr3])].append(cdr3)
    buckets = sorted(pools)
    for b in buckets:
        pools[b] = rng.permutation(np.array(pools[b], dtype=object))
    cursor = {b: 0 for b in buckets}

    def take(b, blocked, tries):
        """One receptor from bucket b that is not blocked, or None."""
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
            got = 0
            for _ in range(k):
                c = take(b, blocked, 60)
                if c is None:
                    # nearest bucket with anything left, outward from b
                    for d in range(1, len(buckets) + 1):
                        for alt in ((b - d, b + d)
                                    if rng.random() < 0.5
                                    else (b + d, b - d)):
                            if alt in pools:
                                c = take(alt, blocked, 60)
                                if c is not None:
                                    break
                        if c is not None:
                            break
                    if c is not None:
                        fallbacks += 1
                if c is None:
                    misses += 1
                    continue
                neg_rows.append((ep, c))
                got += 1

    neg = pd.DataFrame(neg_rows, columns=["epitope", "cdr3"])
    neg["label"] = 0
    out = pd.concat([pos.assign(label=1), neg], ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    if verbose:
        print(f"    {len(out):,} pairs, {100 * out.label.mean():.2f}% positive, "
              f"{fallbacks:,} bucket fallbacks, {misses:,} unplaced")
    return out


def measure(built, label, min_per_class=3):
    """home_max channel strength on one built partition."""
    pos = built[built.label == 1][["epitope", "cdr3"]]
    hm = home_max_map(pos)
    v = built["cdr3"].map(hm).fillna(0.0).to_numpy(dtype=float)
    y = built["label"].to_numpy().astype(np.int64)
    eps = built["epitope"].to_numpy()

    aucs, gaps, npos = [], [], []
    for ep, idx in pd.Series(eps).groupby(eps).indices.items():
        yl = y[idx]
        p, n = int(yl.sum()), int(len(yl) - yl.sum())
        if p < min_per_class or n < min_per_class:
            continue
        aucs.append(auc(yl, v[idx]))
        gaps.append(float(v[idx][yl == 1].mean() - v[idx][yl == 0].mean()))
        npos.append(p)
    aucs, gaps = np.array(aucs), np.array(gaps)

    r = {
        "label": label,
        "n_epitopes": int(len(aucs)),
        "home_max_macro": round(float(aucs.mean()), 4),
        "home_max_pooled": round(auc(y, v), 4),
        "median_gap": round(float(np.median(gaps)), 1),
        "positive_rate": round(float(built.label.mean()), 4),
    }
    print(f"    {label:<10} home_max macro {r['home_max_macro']:.4f}   "
          f"pooled {r['home_max_pooled']:.4f}   "
          f"median gap {r['median_gap']:>+9.1f}   "
          f"({r['n_epitopes']} epitopes)")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-rows", type=int, default=300016)
    ap.add_argument("--min-distance", type=int, default=3)
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = subsample(load_pairs(args.csv), args.max_rows, seed=0)
    print(f"loaded {len(df):,} pairs\n")

    results = {}
    for seed in args.seeds:
        print(f"{'=' * 72}\nSEED {seed}\n{'=' * 72}")
        tr, va, te = epitope_disjoint_split(df, seed=seed)
        r = results.setdefault(f"seed{seed}", {"seed": seed})

        for name, part in (("train", tr), ("test", te)):
            print(f"\n  [{name}] {len(part):,} rows, "
                  f"{part.epitope.nunique()} epitopes")
            print("    building current ...", flush=True)
            cur = tcr_swap_negatives(part, seed=seed,
                                     min_distance=args.min_distance,
                                     verbose=False)
            print("    building matched ...", flush=True)
            fix = tcr_swap_negatives_matched(part, seed=seed,
                                             min_distance=args.min_distance)
            print()
            r[f"{name}_current"] = measure(cur, "current", args.min_per_class)
            r[f"{name}_matched"] = measure(fix, "matched", args.min_per_class)
        print()

    print(f"{'=' * 72}\nVERDICT\n{'=' * 72}")
    cur = [v["test_current"]["home_max_macro"] for v in results.values()]
    fix = [v["test_matched"]["home_max_macro"] for v in results.values()]
    print(f"\ntest partition, home_max per-epitope AUROC")
    print(f"  current  {st.mean(cur):.4f} +- {st.pstdev(cur):.4f}   "
          f"({', '.join(f'{x:.4f}' for x in cur)})")
    print(f"  matched  {st.mean(fix):.4f} +- {st.pstdev(fix):.4f}   "
          f"({', '.join(f'{x:.4f}' for x in fix)})")

    rates = [v["test_matched"]["positive_rate"] for v in results.values()]
    print(f"\n  matched positive rate {st.mean(rates):.4f} "
          f"(must stay ~0.5000)")

    print()
    if abs(st.mean(fix) - 0.5) < 0.05 and abs(st.mean(rates) - 0.5) < 0.01:
        print("CHANNEL CLOSED. Matching the home_max distribution removes it")
        print("while preserving the 50/50 balance and the blocking constraints.")
        print()
        print("Next: port tcr_swap_negatives_matched into tcr_swap.py, rebuild")
        print("the pkls, and re-run. The receptor-only baseline should come back")
        print("to ~0.50 macro, which means macro can be quoted against 0.5")
        print("directly and the debiased dataset in §9.5 becomes defensible as a")
        print("benchmark rather than carrying a known artifact.")
    elif abs(st.mean(rates) - 0.5) >= 0.01:
        print("Balance broke -- too many slots went unplaced under the bucket")
        print("quotas. Widen the buckets or loosen the fallback before reading")
        print("the AUROC column.")
    else:
        print(f"Channel only partly closed ({st.mean(fix):.4f}). Bucketing on")
        print("log2(home_max) is too coarse, or home_max is a proxy for")
        print("something else that survives the match. Report the measured")
        print("baseline instead of fixing the construction.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()