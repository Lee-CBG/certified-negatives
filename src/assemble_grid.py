#!/usr/bin/env python3
"""Assemble the full seen/unseen grid with per-test-set nulls and per-seed verdicts.

The null is the protocol's own: random per-receptor scores, min_per_class=3,
computed on the actual test partition of the pkl each cell was trained on. It
depends only on (epitope, receptor, label) structure, so one null serves every
model and representation sharing that pkl's structure.

    python3 assemble_grid.py                 # uses results/null_cache.json
    python3 assemble_grid.py --refresh-nulls # recompute (slow, loads pkls)
"""
import argparse, json, os, statistics as st, sys
import numpy as np

try:
    from scipy.stats import rankdata
except ImportError:
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

MIN_PC, NDRAW = 3, 500

# cell -> (results json template, pkl template, split label)
# {s} is the seed. Edit here if a filename differs.
CELLS = {
 ("Original",     "seen"):   ("results/baseG_{mode}_{s}.json",        "data/baseg_seed{s}.pkl"),
 ("Original",     "unseen"): ("results/baseU_{mode}_{s}.json",        "data/base_seed{s}.pkl"),
 ("Swap",         "seen"):   ("results/grouped_s{s}_{mode}.json",     "data/grouped_seed{s}.pkl"),
 ("Matched s=1",  "seen"):   ("results/cmp4_s1_{smode}{suf}.json",    "data/mg_seed{s}.pkl"),
 ("Matched s=1",  "unseen"): ("results/s1U_{mode}_{s}.json",          "data/matched_seed{s}.pkl"),
 ("Matched s=4",  "seen"):   ("results/cmp4_s4_{smode}{suf}.json",    "data/m4g_seed{s}.pkl"),
 ("Matched s=4",  "unseen"): ("results/unseen_s4_{smode}_{s}.json",   "data/m4_seed{s}.pkl"),
 ("ESM-2 650M s=4",  "seen"):("results/esm650_m4g_{mode}_{s}.json",   "data/m4g_seed{s}.pkl"),
 ("ESM-2 650M swap", "seen"):("results/esm650_grouped_{mode}_{s}.json","data/grouped_seed{s}.pkl"),
 ("TCR-BERT s=4",    "seen"):("results/tcrbert_m4g_{mode}_{s}.json",  "data/m4g_seed{s}.pkl"),
 ("TCR-BERT swap",   "seen"):("results/tcrbert_grouped_{mode}_{s}.json","data/grouped_seed{s}.pkl"),
}
# seed-0 PLM pilots were written under older names
ALIASES = {
 "results/esm650_m4g_full_0.json":    "results/esm650G_full_0.json",
 "results/esm650_m4g_tcr_only_0.json":"results/esm650G_tcr_only_0.json",
 "results/tcrbert_m4g_full_0.json":   "results/tcrbertG_full_0.json",
 "results/tcrbert_m4g_tcr_only_0.json":"results/tcrbertG_tcr_only_0.json",
 "results/baseU_full_0.json":         "results/base_full.json",
 "results/baseU_tcr_only_0.json":     "results/base_tcr_only.json",
}


def jpath(tmpl, mode, s):
    """cmp4_* used an idiosyncratic naming: seed 0 has no suffix, tcr_only is 'tcr'."""
    smode = "tcr" if mode == "tcr_only" else mode
    suf = "" if s == 0 else f"_{s}"
    p = tmpl.format(mode=mode, smode=smode, suf=suf, s=s)
    if not os.path.exists(p) and p in ALIASES and os.path.exists(ALIASES[p]):
        return ALIASES[p]
    return p


def auc(y, sc):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 < 1 or n0 < 1: return np.nan
    r = rankdata(sc)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def compute_null(pkl):
    import pandas as pd
    df = pd.read_pickle(pkl)
    par = next((c for c in df.columns if "part" in c.lower() or "split" in c.lower()), None)
    te = df[df[par].astype(str).str.lower().str.contains("test")]
    y = te["binding"].to_numpy().astype(int)
    e = te["epi"].astype(str).to_numpy()
    r = te["tcr"].astype(str).to_numpy()
    del df
    groups = []
    for ep in np.unique(e):
        i = np.where(e == ep)[0]; yy = y[i]
        if yy.sum() >= MIN_PC and (len(yy) - yy.sum()) >= MIN_PC:
            groups.append((i, yy))
    uniq, inv = np.unique(r, return_inverse=True)
    rng = np.random.default_rng(0)
    vals = np.empty(NDRAW)
    for b in range(NDRAW):
        sc = rng.random(len(uniq))[inv]
        vals[b] = np.mean([auc(yy, sc[i]) for i, yy in groups])
    return {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
            "n_epitopes": len(groups), "n_test": int(len(te)),
            "n_receptors": int(len(uniq))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-nulls", action="store_true")
    ap.add_argument("--cache", default="results/null_cache.json")
    a = ap.parse_args()

    cache = {}
    if os.path.exists(a.cache) and not a.refresh_nulls:
        cache = json.load(open(a.cache))

    rows, missing, needed = [], [], set()
    for (con, split), (jt, pt) in CELLS.items():
        cell = {"construction": con, "split": split, "seeds": {}}
        for s in (0, 1, 2):
            got = {}
            for mode in ("full", "tcr_only"):
                p = jpath(jt, mode, s)
                if os.path.exists(p):
                    got[mode] = json.load(open(p))
                else:
                    missing.append(p)
            if got: cell["seeds"][s] = got
            if got: needed.add(pt.format(s=s))
        if cell["seeds"]: rows.append(cell)

    for pkl in sorted(needed):
        k = os.path.basename(pkl)
        if k in cache: continue
        if not os.path.exists(pkl):
            print(f"  ! pkl absent, no null: {pkl}"); continue
        print(f"  computing null for {k} ...", flush=True)
        cache[k] = compute_null(pkl)
    os.makedirs(os.path.dirname(a.cache) or ".", exist_ok=True)
    json.dump(cache, open(a.cache, "w"), indent=2)

    print(f"\n{'='*118}\nPER SEED, macro AUROC per epitope\n")
    hdr = (f"{'construction':17s} {'split':7s} {'sd':>3s} {'full':>7s} {'tcr':>7s} "
           f"{'contrib':>8s} {'nullSD':>7s} {'z_tcr':>7s} {'K':>5s}  verdict")
    print(hdr); print("-" * len(hdr))
    agg = {}
    for cell in rows:
        con, split = cell["construction"], cell["split"]
        pt = CELLS[(con, split)][1]
        for s, got in sorted(cell["seeds"].items()):
            nk = os.path.basename(pt.format(s=s))
            nul = cache.get(nk)
            f = got.get("full", {}).get("macro_auroc_per_epitope")
            t = got.get("tcr_only", {}).get("macro_auroc_per_epitope")
            con_v = (f - t) if (f is not None and t is not None) else None
            if nul and t is not None:
                z = (t - nul["mean"]) / nul["sd"]
                verdict = "pass" if abs(t - 0.500) <= nul["sd"] else "FAIL"
                nsd, K = nul["sd"], nul["n_epitopes"]
            else:
                z, verdict, nsd, K = None, "no null", None, None
            print(f"{con:17s} {split:7s} {s:3d} "
                  f"{(f if f is not None else float('nan')):7.4f} "
                  f"{(t if t is not None else float('nan')):7.4f} "
                  f"{(con_v if con_v is not None else float('nan')):+8.4f} "
                  f"{(nsd if nsd else float('nan')):7.5f} "
                  f"{(z if z is not None else float('nan')):+7.2f} "
                  f"{(K if K else -1):5d}  {verdict}")
            agg.setdefault((con, split), {"full": [], "tcr": [], "con": [], "verdicts": []})
            if f is not None: agg[(con, split)]["full"].append(f)
            if t is not None: agg[(con, split)]["tcr"].append(t)
            if con_v is not None: agg[(con, split)]["con"].append(con_v)
            agg[(con, split)]["verdicts"].append(verdict)

    print(f"\n{'='*118}\nGRID, mean +/- SAMPLE sd over seeds\n")
    h2 = (f"{'construction':17s} {'split':7s} {'full':>17s} {'tcr_only':>17s} "
          f"{'contribution':>19s}  certified")
    print(h2); print("-" * len(h2))
    for (con, split), d in agg.items():
        def ms(x):
            if not x: return "       n/a       "
            if len(x) == 1: return f"{x[0]:.4f} (n=1)     "
            return f"{st.mean(x):.4f} +/- {st.stdev(x):.4f}"
        cert = ("yes" if d["verdicts"] and all(v == "pass" for v in d["verdicts"])
                else ("NO (" + ",".join(v for v in d["verdicts"]) + ")"
                      if "FAIL" in d["verdicts"] else "unknown"))
        print(f"{con:17s} {split:7s} {ms(d['full']):>17s} {ms(d['tcr']):>17s} "
              f"{ms(d['con']):>19s}  {cert}")

    if missing:
        print(f"\n{len(missing)} result files not yet present:")
        for p in sorted(set(missing)): print(f"  {p}")
    print(f"\nnull cache: {a.cache}")


if __name__ == "__main__":
    main()