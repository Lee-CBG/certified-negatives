#!/usr/bin/env python3
"""What does a receptor-only model's score actually track?

In tcr_only mode the epitope is zeroed, so the score is a deterministic
function of the receptor and every unique receptor carries one score. This
correlates that score directly against construction covariates over all unique
test receptors, which is far better powered than a per-epitope analysis, and
reports partial correlations so draw count and home size can be separated.

    python3 receptor_channel.py --pkl data/m4g_seed0.pkl \
        --preds BLOSUM=results/cmp4_s4_tcr.npz \
        --preds ESM2=results/esm650G_tcr_only_0.npz \
        --preds TCRBERT=results/tcrbertG_tcr_only_0.npz
"""
import argparse, os
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


def partial(x, y, z):
    """Spearman partial correlation of x,y controlling for z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    ex, ey = resid(rx, rz), resid(ry, rz)
    d = np.sqrt((ex @ ex) * (ey @ ey))
    return float(ex @ ey / d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--preds", action="append", required=True)
    a = ap.parse_args()

    df = pd.read_pickle(a.pkl)
    par = next(c for c in df.columns if "part" in c.lower() or "split" in c.lower())
    epi = df["epi"].astype(str).to_numpy()
    tcr = df["tcr"].astype(str).to_numpy()
    lab = df["binding"].to_numpy().astype(int)
    part = df[par].astype(str).to_numpy()
    del df

    pos = lab == 1
    ep_size = pd.Series(epi[pos]).value_counts()
    hm = pd.DataFrame({"tcr": tcr[pos], "epi": epi[pos]})
    hm["sz"] = hm["epi"].map(ep_size)
    home = hm.groupby("tcr")["sz"].max()
    draw = pd.Series(tcr[~pos]).value_counts()
    nrow = pd.Series(tcr).value_counts()
    npos = pd.Series(tcr[pos]).value_counts()

    te = part == "test"
    t, y, e = tcr[te], lab[te], epi[te]
    print(f"{os.path.basename(a.pkl)}: test {te.sum():,} rows, "
          f"{len(np.unique(t)):,} unique receptors")

    for spec in a.preds:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"\n{name}: MISSING {path}"); continue
        z = np.load(path, allow_pickle=True)
        ks = next((k for k in z.files if "score" in k.lower() or "pred" in k.lower()), None)
        ke = next((k for k in z.files if "epi" in k.lower()), None)
        sc = np.asarray(z[ks], float).ravel()
        if len(sc) != te.sum():
            print(f"\n{name}: length mismatch, skipping"); continue
        if ke is not None and not np.array_equal(np.asarray(z[ke]).astype(str).ravel(), e):
            print(f"\n{name}: row order differs from pkl test frame, skipping"); continue

        # collapse to unique receptors; verify the ablation really is receptor-only
        g = pd.DataFrame({"tcr": t, "s": sc}).groupby("tcr")["s"]
        spread = float((g.max() - g.min()).max())
        s_r = g.mean()
        keys = s_r.index.to_numpy()
        cov = pd.DataFrame({
            "home":  home.reindex(keys).fillna(1).to_numpy(),
            "draw":  draw.reindex(keys).fillna(0).to_numpy(),
            "nrow":  nrow.reindex(keys).fillna(1).to_numpy(),
            "npos":  npos.reindex(keys).fillna(0).to_numpy(),
            "len":   np.array([len(k) for k in keys], float),
        })
        sv = s_r.to_numpy()
        print(f"\n{name}: {len(keys):,} unique receptors, "
              f"max within-receptor score spread {spread:.2e} "
              f"({'receptor-only confirmed' if spread < 1e-5 else 'NOT a pure function of the receptor'})")
        print(f"  {'covariate':9s} {'spearman':>10s} {'partial|home':>14s}")
        for c in cov.columns:
            if spearmanr is None:
                continue
            rho, p = spearmanr(cov[c].to_numpy(), sv)
            star = "*" if p < 0.001 else ("." if p < 0.05 else " ")
            if c == "home":
                print(f"  {c:9s} {rho:+10.4f}{star} {'-':>14s}")
            else:
                pc = partial(cov[c].to_numpy(), sv, cov["home"].to_numpy())
                print(f"  {c:9s} {rho:+10.4f}{star} {pc:+14.4f}")
    print("\n* p<0.001, . p<0.05.  n is receptors, so power is ~75x the per-epitope test.")


if __name__ == "__main__":
    main()