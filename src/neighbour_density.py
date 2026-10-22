#!/usr/bin/env python3
"""Does the receptor-only score track sequence isolation?

The draw blocks any donor within Levenshtein distance d of one of the epitope's
binders, so negatives should be systematically more isolated in sequence space.
A language-model embedding can represent that; a per-position substitution
profile largely cannot. This counts each test receptor's neighbours within edit
distance <= d among all receptors in the file, then correlates that count
against the score, partialling out draw count so isolation is separated from
draw frequency.

Exact Levenshtein is pruned three ways: length bucketing (|len diff| <= d),
a character-multiset bound, and early-abort banded DP. Still the expensive
part, so --sample caps how many test receptors are scored.

    python3 neighbour_density.py --pkl data/m4g_seed0.pkl --d 3 --sample 4000 \
        --preds BLOSUM=results/cmp4_s4_tcr.npz \
        --preds ESM2=results/esm650G_tcr_only_0.npz \
        --preds TCRBERT=results/tcrbertG_tcr_only_0.npz
"""
import argparse, json, os, sys, time
from collections import Counter
import numpy as np
import pandas as pd

try:
    from scipy.stats import rankdata, spearmanr
except ImportError:
    spearmanr = None
    def rankdata(a):
        a = np.asarray(a); n = len(a)
        o = np.argsort(a, kind="mergesort")
        r = np.empty(n, float); r[o] = np.arange(1, n + 1)
        s = a[o]; i = 0
        while i < n:
            j = i
            while j + 1 < n and s[j + 1] == s[i]: j += 1
            if j > i: r[o[i:j + 1]] = (i + 1 + j + 1) / 2.0
            i = j + 1
        return r


def bounded_lev(a, b, k):
    """True if edit distance(a,b) <= k. Full DP, early abort on row minimum."""
    la, lb = len(a), len(b)
    if abs(la - lb) > k:
        return False
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        ca = a[i - 1]
        cur = [i] + [0] * lb
        best = i
        for j in range(1, lb + 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != b[j - 1]))
            cur[j] = v
            if v < best:
                best = v
        if best > k:
            return False
        prev = cur
    return prev[lb] <= k


def partial(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    ex, ey = resid(rx, rz), resid(ry, rz)
    den = np.sqrt((ex @ ex) * (ey @ ey))
    return float(ex @ ey / den) if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--preds", action="append", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()

    df = pd.read_pickle(a.pkl)
    par = next(c for c in df.columns if "part" in c.lower() or "split" in c.lower())
    epi = df["epi"].astype(str).to_numpy()
    tcr = df["tcr"].astype(str).to_numpy()
    lab = df["binding"].to_numpy().astype(int)
    part = df[par].astype(str).to_numpy()
    del df

    universe = np.unique(tcr)
    by_len = {}
    for s in universe:
        by_len.setdefault(len(s), []).append(s)
    by_len = {L: (v, [Counter(x) for x in v]) for L, v in by_len.items()}
    print(f"{os.path.basename(a.pkl)}: {len(universe):,} receptors in the file, "
          f"lengths {min(by_len)} to {max(by_len)}")

    te = part == "test"
    t, y, e = tcr[te], lab[te], epi[te]
    keys_all = np.unique(t)
    rng = np.random.default_rng(a.seed)
    keys = (rng.choice(keys_all, a.sample, replace=False)
            if a.sample and a.sample < len(keys_all) else keys_all)
    print(f"test {te.sum():,} rows, {len(keys_all):,} unique receptors, "
          f"scoring {len(keys):,}")

    nbr = {}
    if a.cache and os.path.exists(a.cache):
        nbr = dict(json.load(open(a.cache)))
        print(f"loaded {len(nbr):,} cached neighbour counts")
    todo = [k for k in keys if k not in nbr]
    t0 = time.time()
    for i, s in enumerate(todo):
        cs, n = Counter(s), 0
        for L in range(len(s) - a.d, len(s) + a.d + 1):
            if L not in by_len: continue
            cands, ccs = by_len[L]
            for b, cb in zip(cands, ccs):
                if b == s: continue
                # character-multiset lower bound on edit distance
                diff = sum((cs - cb).values()) + sum((cb - cs).values())
                if diff > 2 * a.d: continue
                if bounded_lev(s, b, a.d): n += 1
        nbr[s] = n
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"    {i+1:,}/{len(todo):,}  {el:.0f}s  "
                  f"eta {el/(i+1)*(len(todo)-i-1):.0f}s", flush=True)
    if a.cache:
        json.dump(nbr, open(a.cache, "w"))
        print(f"wrote neighbour cache {a.cache} ({len(nbr):,} receptors)")
    nv = np.array([nbr[k] for k in keys], float)
    print(f"neighbours within d<={a.d}: median {np.median(nv):.0f}, "
          f"mean {nv.mean():.1f}, max {nv.max():.0f}, "
          f"zero-neighbour {int((nv==0).sum()):,}/{len(nv):,}")

    # is isolation actually associated with the label, as the blocking rule implies?
    isneg = pd.DataFrame({"tcr": t, "y": y}).groupby("tcr")["y"].mean().reindex(keys).to_numpy()
    if spearmanr is not None:
        rho, p = spearmanr(nv, isneg)
        print(f"neighbours vs receptor positive-rate: rho={rho:+.4f} p={p:.2e}")

    draw = pd.Series(tcr[lab == 0]).value_counts().reindex(keys).fillna(0).to_numpy()
    print(f"\n{'representation':14s} {'rho(nbr,score)':>16s} {'partial|draw':>14s}")
    for spec in a.preds:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"{name:14s} MISSING"); continue
        z = np.load(path, allow_pickle=True)
        ks = next((k for k in z.files if "score" in k.lower() or "pred" in k.lower()), None)
        ke = next((k for k in z.files if "epi" in k.lower()), None)
        sc = np.asarray(z[ks], float).ravel()
        if len(sc) != te.sum(): print(f"{name:14s} length mismatch"); continue
        if ke is not None and not np.array_equal(np.asarray(z[ke]).astype(str).ravel(), e):
            print(f"{name:14s} row order differs"); continue
        sr = pd.DataFrame({"tcr": t, "s": sc}).groupby("tcr")["s"].mean().reindex(keys).to_numpy()
        if spearmanr is None: continue
        rho, p = spearmanr(nv, sr)
        star = "*" if p < 0.001 else ("." if p < 0.05 else " ")
        print(f"{name:14s} {rho:+16.4f}{star} {partial(nv, sr, draw):+14.4f}")
    print("\n* p<0.001, . p<0.05")
    print("Hypothesis: isolation is read by the PLMs and not by BLOSUM.")


if __name__ == "__main__":
    main()