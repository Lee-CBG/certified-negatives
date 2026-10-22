#!/usr/bin/env python3
"""Is sequence isolation the channel the language models read?

Computes exact neighbour counts for every receptor with rapidfuzz, then asks
the two questions that matter, on full test sets rather than a sample:

  1. CONSTRUCTION. Within each epitope, do negatives differ from positives in
     neighbour count? A null global correlation can still hide a within-epitope
     gap; that is how home size and draw count behaved.
  2. MODEL. Does per-epitope receptor-only AUROC track that gap, and does it do
     so for the language models and not for BLOSUM?

Also repeats the receptor-level partials at full n.

    python3 isolation_test.py --pkls data/m4g_seed0.pkl data/m4g_seed1.pkl data/m4g_seed2.pkl \
        --d 3 --cache results/nbr_all_d3.npz
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

MIN_PC = 3


def auc(y, sc):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 < 1 or n0 < 1: return np.nan
    r = rankdata(sc)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def partial(x, y, Z):
    rx, ry = rankdata(x), rankdata(y)
    B = np.c_[np.ones(len(x)), *[rankdata(z) for z in Z]]
    r = lambda a: a - B @ np.linalg.lstsq(B, a, rcond=None)[0]
    ex, ey = r(rx), r(ry)
    d = np.sqrt((ex @ ex) * (ey @ ey))
    return float(ex @ ey / d) if d else float("nan")


def star(p):
    return "*" if p < 1e-3 else ("." if p < 0.05 else " ")


def neighbour_table(universe, d, cache, chunk=5000):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if np.array_equal(z["keys"], universe) and int(z["d"]) == d:
            print(f"loaded exact neighbour table from {cache}")
            return z["counts"]
        print(f"{cache} does not match this universe or d; recomputing")
    from rapidfuzz import process as rp, distance as rd
    out = np.zeros(len(universe), dtype=np.int32)
    t0 = time.time()
    for i in range(0, len(universe), chunk):
        q = universe[i:i + chunk]
        m = rp.cdist(q, universe, scorer=rd.Levenshtein.distance,
                     score_cutoff=d, dtype=np.uint8, workers=-1)
        out[i:i + chunk] = (m <= d).sum(1) - 1          # exclude self
        del m
        print(f"  {min(i+chunk, len(universe)):,}/{len(universe):,}  "
              f"{time.time()-t0:.0f}s", flush=True)
    if cache:
        np.savez_compressed(cache, keys=universe, counts=out, d=d)
        print(f"wrote {cache}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkls", nargs="+", required=True)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--cache", default="results/nbr_all_d3.npz")
    a = ap.parse_args()

    frames = []
    for p in a.pkls:
        df = pd.read_pickle(p)
        par = next(c for c in df.columns if "part" in c.lower() or "split" in c.lower())
        frames.append(dict(
            pkl=p,
            tcr=df["tcr"].astype(str).to_numpy(),
            epi=df["epi"].astype(str).to_numpy(),
            lab=df["binding"].to_numpy().astype(int),
            part=df[par].astype(str).to_numpy()))
        del df
    universe = np.unique(np.concatenate([f["tcr"] for f in frames]))
    print(f"{len(universe):,} distinct receptors across {len(frames)} files")
    counts = neighbour_table(universe, a.d, a.cache)
    nbr = dict(zip(universe.tolist(), counts.tolist()))
    print(f"neighbours within d<={a.d}: median {np.median(counts):.0f} "
          f"mean {counts.mean():.1f} max {counts.max()} "
          f"zero {int((counts==0).sum()):,}/{len(counts):,}\n")

    for fi, f in enumerate(frames):
        seed = fi
        tcr, epi, lab, part = f["tcr"], f["epi"], f["lab"], f["part"]
        pos = lab == 1
        ep_size = pd.Series(epi[pos]).value_counts()
        hm = pd.DataFrame({"tcr": tcr[pos], "epi": epi[pos]})
        hm["sz"] = hm["epi"].map(ep_size)
        home = hm.groupby("tcr")["sz"].max()
        draw = pd.Series(tcr[~pos]).value_counts()

        te = part == "test"
        t, y, e = tcr[te], lab[te], epi[te]
        nv_row = np.array([nbr[s] for s in t], float)
        ln_row = np.array([len(s) for s in t], float)

        rows, gaps = [], []
        for ep in np.unique(e):
            m = e == ep
            yy = y[m]
            if yy.sum() < MIN_PC or (len(yy) - yy.sum()) < MIN_PC: continue
            p_, q_ = yy == 1, yy == 0
            g = float(np.log1p(nv_row[m][q_]).mean() - np.log1p(nv_row[m][p_]).mean())
            rows.append(dict(idx=np.where(m)[0], y=yy, nbr_gap=g,
                             home_gap=float(np.log1p(home.reindex(t[m][q_]).fillna(1)).mean()
                                            - np.log1p(home.reindex(t[m][p_]).fillna(1)).mean()),
                             draw_gap=float(np.log1p(draw.reindex(t[m][q_]).fillna(0)).mean()
                                            - np.log1p(draw.reindex(t[m][p_]).fillna(0)).mean())))
            gaps.append(g)
        gaps = np.array(gaps)
        print(f"##### {os.path.basename(f['pkl'])}: {len(rows)} epitopes qualify")
        sd = gaps.std(ddof=1); se = sd / np.sqrt(len(gaps))
        print(f"  [1] within-epitope neighbour gap, log1p(neg) - log1p(pos)")
        print(f"      mean {gaps.mean():+.4f}  sd {sd:.4f}  se {se:.4f}  "
              f"t={gaps.mean()/se:+.2f}   {'SYSTEMATIC' if abs(gaps.mean()/se) > 3 else 'no systematic gap'}")

        P = ({0: ("results/cmp4_s4_tcr.npz", "results/esm650G_tcr_only_0.npz",
                  "results/tcrbertG_tcr_only_0.npz")}
             .get(seed, (f"results/cmp4_s4_tcr_{seed}.npz",
                         f"results/esm650_m4g_tcr_only_{seed}.npz",
                         f"results/tcrbert_m4g_tcr_only_{seed}.npz")))
        print(f"  [2] per-epitope AUROC vs gaps, and [3] receptor-level, full n")
        print(f"      {'rep':8s} {'macro':>7s} {'rho(nbrgap)':>12s} {'rho(drawgap)':>13s} "
              f"{'rho(nbr,score)':>15s} {'partial|draw,len':>17s}")
        for name, path in zip(("BLOSUM", "ESM2", "TCRBERT"), P):
            if not os.path.exists(path):
                print(f"      {name:8s} MISSING {path}"); continue
            z = np.load(path, allow_pickle=True)
            ks = next(k for k in z.files if "score" in k.lower() or "pred" in k.lower())
            ke = next((k for k in z.files if "epi" in k.lower()), None)
            sc = np.asarray(z[ks], float).ravel()
            if len(sc) != te.sum() or (ke is not None and
                    not np.array_equal(np.asarray(z[ke]).astype(str).ravel(), e)):
                print(f"      {name:8s} row order or length mismatch"); continue
            aucs = np.array([auc(r["y"], sc[r["idx"]]) for r in rows])
            r1, p1 = spearmanr([r["nbr_gap"] for r in rows], aucs)
            r2, p2 = spearmanr([r["draw_gap"] for r in rows], aucs)
            sr = pd.DataFrame({"tcr": t, "s": sc}).groupby("tcr")["s"].mean()
            keys = sr.index.to_numpy()
            nvk = np.array([nbr[k] for k in keys], float)
            lnk = np.array([len(k) for k in keys], float)
            dwk = draw.reindex(keys).fillna(0).to_numpy()
            svk = sr.to_numpy()
            r3, p3 = spearmanr(nvk, svk)
            pc = partial(nvk, svk, [dwk, lnk])
            print(f"      {name:8s} {np.nanmean(aucs):7.4f} {r1:+11.3f}{star(p1)} "
                  f"{r2:+12.3f}{star(p2)} {r3:+14.3f}{star(p3)} {pc:+17.3f}")
        print()
    print("* p<0.001, . p<0.05.  [1] is construction-level and model-free.")


if __name__ == "__main__":
    main()