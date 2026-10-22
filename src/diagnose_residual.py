#!/usr/bin/env python3
"""Why does a richer representation read a residual that BLOSUM62 cannot?

For one construction, computes per-epitope tcr_only AUROC for each available
representation and correlates it against construction covariates derived from
the pkl itself. The prediction under PROJECT_NOTES_SCALE4 4.15 is that the PLM
residual tracks draw count where BLOSUM's does not.

    python3 diagnose_residual.py --pkl data/m4g_seed0.pkl --scale 4 \
        --preds BLOSUM=results/cmp4_s4_tcr.npz \
        --preds ESM2=results/esm650_m4g_tcr_only_0.npz \
        --preds TCRBERT=results/tcrbert_m4g_tcr_only_0.npz
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

MIN_PC = 3


def auc(y, sc):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 < 1 or n0 < 1: return np.nan
    r = rankdata(sc)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--preds", action="append", required=True,
                    help="NAME=path.npz ; repeat")
    a = ap.parse_args()

    df = pd.read_pickle(a.pkl)
    par = next(c for c in df.columns if "part" in c.lower() or "split" in c.lower())
    epi_all = df["epi"].astype(str).to_numpy()
    tcr_all = df["tcr"].astype(str).to_numpy()
    lab_all = df["binding"].to_numpy().astype(int)
    part = df[par].astype(str).to_numpy()
    del df

    # ---- global receptor covariates, computed over the whole file -----------
    pos = lab_all == 1
    ep_size = pd.Series(epi_all[pos]).value_counts()            # positives per epitope
    hm = pd.DataFrame({"tcr": tcr_all[pos], "epi": epi_all[pos]})
    hm["sz"] = hm["epi"].map(ep_size)
    home = hm.groupby("tcr")["sz"].max()                         # h(r)
    draw = pd.Series(tcr_all[~pos]).value_counts()               # times used as negative
    nrow = pd.Series(tcr_all).value_counts()                     # total rows per receptor
    print(f"{a.pkl}: {len(epi_all):,} rows, {len(home):,} receptors with a home, "
          f"{len(ep_size):,} epitopes with positives")

    te = part == "test"
    e, t, y = epi_all[te], tcr_all[te], lab_all[te]
    h = home.reindex(t).fillna(1).to_numpy()
    d = draw.reindex(t).fillna(0).to_numpy()
    n = nrow.reindex(t).fillna(1).to_numpy()

    # ---- per-epitope covariates --------------------------------------------
    rows = []
    for ep in np.unique(e):
        m = e == ep
        yy = y[m]
        if yy.sum() < MIN_PC or (len(yy) - yy.sum()) < MIN_PC: continue
        p, q = yy == 1, yy == 0
        sz = float(ep_size.get(ep, np.nan))
        lg = a.scale * np.log2(max(sz, 1.0))
        rows.append(dict(
            epitope=ep, idx=np.where(m)[0], y=yy, size=sz,
            bucket_pos=lg - np.floor(lg),
            home_gap=float(np.log2(h[m][q] + 1).mean() - np.log2(h[m][p] + 1).mean()),
            draw_gap=float(np.log2(d[m][q] + 1).mean() - np.log2(d[m][p] + 1).mean()),
            rows_gap=float(np.log2(n[m][q] + 1).mean() - np.log2(n[m][p] + 1).mean()),
        ))
    print(f"{len(rows)} epitopes qualify at min_per_class={MIN_PC}\n")

    covs = ["size", "bucket_pos", "home_gap", "draw_gap", "rows_gap"]
    print(f"{'representation':14s} {'macro':>7s} {'perEpSD':>8s}  " +
          "  ".join(f"{c:>10s}" for c in covs))
    print("-" * (14 + 7 + 8 + 4 + 12 * len(covs)))

    for spec in a.preds:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"{name:14s} MISSING {path}"); continue
        z = np.load(path, allow_pickle=True)
        ks = next((k for k in z.files if "score" in k.lower() or "pred" in k.lower()), None)
        ke = next((k for k in z.files if "epi" in k.lower()), None)
        sc = np.asarray(z[ks], float).ravel()
        if len(sc) != len(e):
            print(f"{name:14s} length {len(sc)} != test rows {len(e)}, skipping")
            continue
        if ke is not None:
            ne = np.asarray(z[ke]).astype(str).ravel()
            if not np.array_equal(ne, e):
                print(f"{name:14s} ROW ORDER differs from the pkl test frame, skipping "
                      f"(first mismatch at {int(np.argmax(ne != e))})")
                continue
        if "label" in "".join(z.files).lower():
            kl = next(k for k in z.files if "label" in k.lower())
            if not np.array_equal(np.asarray(z[kl]).ravel().astype(int), y):
                print(f"{name:14s} LABELS differ from the pkl test frame, skipping")
                continue
        aucs = np.array([auc(r["y"], sc[r["idx"]]) for r in rows])
        line = f"{name:14s} {np.nanmean(aucs):7.4f} {np.nanstd(aucs, ddof=1):8.4f}  "
        for c in covs:
            v = np.array([r[c] for r in rows], float)
            ok = ~np.isnan(v) & ~np.isnan(aucs)
            if spearmanr is not None and ok.sum() > 5:
                rho, p = spearmanr(v[ok], aucs[ok])
                star = "*" if p < 0.01 else ("." if p < 0.05 else " ")
                line += f"{rho:+9.3f}{star} "
            else:
                line += f"{'n/a':>10s} "
        print(line)

    print("\n* p<0.01, . p<0.05 (Spearman, per-epitope)")
    print("Prediction under 4.15: draw_gap correlates for the PLMs and not for BLOSUM.")


if __name__ == "__main__":
    main()