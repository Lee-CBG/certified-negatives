"""Why is a receptor-only model anti-predictive WITHIN epitopes?

THE EFFECT, now firmly established

Receptor-grouped split, zero receptor overlap, 285-290 epitopes scored:

    MLP tcr_only macro   0.4557 / 0.4542 / 0.4376     z = -6.2 / -6.8 / -7.8
    n-gram floor macro   0.4733 / 0.4642 / 0.4475     z = -3.8 / -5.3 / -6.6
    simulated null       0.5000, sd 0.007 (random per-receptor scores)

Two model classes with nothing in common land on ~0.45, so this is a property of
the data rather than of either model. Pooled AUROC is ABOVE chance on the same
runs (0.5101 / 0.5065 / 0.5148), so it is a Simpson reversal: the receptor
channel predicts the label correctly across the dataset and incorrectly inside
each epitope.

HYPOTHESES ALREADY DEAD

    pool weight direction   predicted negatives heavier; positives are heavier
    length                  macro 0.4985 as a standalone predictor
    publicness              macro 0.5098
    donor-pool weight       macro 0.5099
    receptor leverage       random per-receptor scores give 0.5001 +- 0.0008

THE ONE LEFT

`tcr_swap_negatives` draws the donor pool proportional to positive occurrence,
and occurrence is dominated by the largest epitopes -- the top ten hold 45-70% of
it. So the class composition inside an epitope depends on that epitope's size:

    small epitope   positives are its own private binders
                    negatives are overwhelmingly LARGE-epitope binders
    large epitope   positives and negatives are both large-epitope binders

"Home epitope size" -- how big the epitopes are that a receptor actually binds --
is therefore a receptor-level covariate that separates the classes differently in
every epitope, and it has not been tested. Its predicted signature is a strong
class gap in small epitopes fading to nothing in large ones, matching the
observed banding (1-5: 0.44 rising to 101+: 0.49).

HOW THE SCORES ARE ALIGNED

train_pkl.py --save-preds writes scores, labels and epitopes but not receptors.
The test frame is `df[df.partition == "test"].reset_index(drop=True)`, which is
deterministic, so re-reading the pkl and taking the same rows in the same order
recovers the receptors. The epitope arrays are compared elementwise as a guard
before anything is computed -- if the pkl has been rebuilt since the run, that
check fails loudly instead of silently misaligning.

WHAT EACH OUTCOME MEANS

    home size reproduces ~0.45  -> the cause is the pool being occurrence-weighted
        and size-skewed. Fix: draw each epitope's negatives from a pool matched on
        home epitope size, or uniformly over distinct receptors. Rebuild, re-run,
        the baseline should return to 0.5.

    it does not                 -> stop. Five hypotheses is enough. Report the
        anti-predictive receptor baseline as a measured property of swapped
        negatives, quote every macro against the tcr_only condition rather than
        against 0.5, and note it as an open question. The contribution metric
        (full minus tcr_only) is unaffected either way, which is why this does
        not block the writeup.

USAGE

    export PYTHONPATH=src
    python3 src/diagnose_home_size.py --out results/home_size.json

CPU only, about a minute per seed.
"""

import argparse
import gc
import json
import os
import statistics as st

import numpy as np
import pandas as pd

SIZE_BINS = [0, 5, 10, 25, 100, 10 ** 9]
SIZE_LABELS = ["1-5", "6-10", "11-25", "26-100", "101+"]


def _avg_ranks(a):
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc(y, s):
    r = _avg_ranks(s)
    npos = float(y.sum())
    nneg = float(len(y) - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def per_epitope(y, s, eps, min_per_class=3):
    out = {}
    for ep, idx in pd.Series(eps).groupby(eps).indices.items():
        yl = y[idx]
        p, n = int(yl.sum()), int(len(yl) - yl.sum())
        if p < min_per_class or n < min_per_class:
            continue
        out[ep] = (auc(yl, s[idx]), p, idx)
    return out


def banded(vals, npos):
    b = np.asarray(pd.cut(npos, SIZE_BINS, labels=SIZE_LABELS), dtype=object)
    return {lab: round(float(np.mean(vals[b == lab])), 4)
            for lab in SIZE_LABELS if (b == lab).any()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pkl-template", default="data/grouped_seed{}.pkl")
    ap.add_argument("--npz-template",
                    default="results/preds_grouped_s{}_tcr_only.npz")
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for seed in args.seeds:
        pkl = args.pkl_template.format(seed)
        npz = args.npz_template.format(seed)
        if not (os.path.exists(pkl) and os.path.exists(npz)):
            print(f"!! seed {seed}: missing {pkl if not os.path.exists(pkl) else npz}")
            continue

        print(f"\n{'=' * 78}\nseed {seed}\n{'=' * 78}")
        d = np.load(npz, allow_pickle=True)
        scores, labels = d["scores"], d["labels"].astype(np.int64)

        df = pd.read_pickle(pkl)
        tcr_col = "tcr" if "tcr" in df.columns else "cdr3"
        epi_col = "epi" if "epi" in df.columns else "epitope"
        lab_col = "binding" if "binding" in df.columns else "label"
        work = df[[tcr_col, epi_col, lab_col, "partition"]].rename(
            columns={tcr_col: "tcr", epi_col: "epi", lab_col: "binding"})
        del df
        gc.collect()

        te = work[work.partition == "test"].reset_index(drop=True)
        if len(te) != len(scores) or not np.array_equal(
                te["epi"].to_numpy().astype(str), d["epitopes"].astype(str)):
            print("  !! saved predictions do not line up with this pkl.")
            print("     The pkl was rebuilt after the run. Re-run training or")
            print("     rebuild is the fix; nothing below would be meaningful.")
            continue
        print(f"  aligned: {len(te):,} rows, {te.epi.nunique()} epitopes")

        # Home epitope size: how big are the epitopes this receptor binds?
        # Computed over the WHOLE pkl from positives only, so it does not depend
        # on the split and is not contaminated by the draw.
        pos_all = work[work.binding == 1]
        epi_size = pos_all["epi"].value_counts()
        home = (pos_all.assign(sz=pos_all["epi"].map(epi_size))
                .groupby("tcr")["sz"])
        te = te.assign(
            home_max=te["tcr"].map(home.max()).fillna(0.0).astype(float),
            home_mean=te["tcr"].map(home.mean()).fillna(0.0).astype(float),
            weight=te["tcr"].map(pos_all["tcr"].value_counts()).fillna(0.0).astype(float),
            length=te["tcr"].str.len().astype(float),
        )

        y = te["binding"].to_numpy().astype(np.int64)
        eps = te["epi"].to_numpy().astype(str)
        model = per_epitope(y, scores, eps, args.min_per_class)
        m_aucs = np.array([v[0] for v in model.values()])
        m_npos = np.array([v[1] for v in model.values()])
        print(f"\n  model tcr_only macro {m_aucs.mean():.4f} "
              f"over {len(m_aucs)} epitopes")

        r = results.setdefault(f"seed{seed}", {"seed": seed})
        r["model_macro"] = round(float(m_aucs.mean()), 4)
        r["model_bands"] = banded(m_aucs, m_npos)

        covs = ["home_max", "home_mean", "weight", "length"]
        print(f"\n  {'covariate':<12}{'macro':>9}{'pooled':>9}{'corr w/ score':>15}"
              f"   " + "".join(f"{b:>9}" for b in SIZE_LABELS))
        for c in covs:
            v = te[c].to_numpy()
            pe = per_epitope(y, v, eps, args.min_per_class)
            a = np.array([x[0] for x in pe.values()])
            n = np.array([x[1] for x in pe.values()])
            cc = float(np.corrcoef(scores, v)[0, 1]) if v.std() > 0 else float("nan")
            bands = banded(a, n)
            r[c] = {"macro": round(float(a.mean()), 4),
                    "pooled": round(auc(y, v), 4),
                    "corr_with_score": round(cc, 4),
                    "bands": bands}
            print(f"  {c:<12}{a.mean():>9.4f}{auc(y, v):>9.4f}{cc:>+15.3f}"
                  f"   " + "".join(f"{bands.get(b, float('nan')):>9.4f}"
                                   for b in SIZE_LABELS))

        # Within-epitope class gap on home size, by epitope size.
        print(f"\n  within-epitope home_max gap (positives minus negatives)")
        gaps, sizes = [], []
        for ep, (a, p, idx) in model.items():
            yl, hv = y[idx], te["home_max"].to_numpy()[idx]
            gaps.append(float(hv[yl == 1].mean() - hv[yl == 0].mean()))
            sizes.append(p)
        gaps, sizes = np.array(gaps), np.array(sizes)
        gb = banded(gaps, sizes)
        r["home_gap_bands"] = gb
        print("    " + "".join(f"{b:>12}" for b in SIZE_LABELS))
        print("    " + "".join(f"{gb.get(b, float('nan')):>12.1f}"
                               for b in SIZE_LABELS))
        cc = float(np.corrcoef(gaps, m_aucs)[0, 1])
        r["corr_gap_vs_model_auroc"] = round(cc, 4)
        print(f"    corr(gap, per-epitope model AUROC) = {cc:+.3f}")

        del work, te, pos_all
        gc.collect()

    # -- verdict ---------------------------------------------------------------
    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    if not results:
        print("No seeds processed.")
        return

    mm = [v["model_macro"] for v in results.values()]
    print(f"\nmodel tcr_only macro to explain: {st.mean(mm):.4f}   "
          f"({', '.join(f'{x:.4f}' for x in mm)})")
    print(f"\n{'covariate':<14}{'macro':>10}{'sd':>9}{'corr w/ score':>15}")
    print("-" * 78)
    best, best_gap = None, 1.0
    for c in ("home_max", "home_mean", "weight", "length"):
        ms = [v[c]["macro"] for v in results.values() if c in v]
        cs = [v[c]["corr_with_score"] for v in results.values() if c in v]
        if not ms:
            continue
        print(f"{c:<14}{st.mean(ms):>10.4f}{st.pstdev(ms):>9.4f}"
              f"{st.mean(cs):>+15.3f}")
        if abs(st.mean(ms) - st.mean(mm)) < best_gap:
            best, best_gap = c, abs(st.mean(ms) - st.mean(mm))

    print()
    if best_gap < 0.02:
        print(f"'{best}' reproduces the model's macro to within {best_gap:.4f}.")
        print("That makes the anti-predictive baseline a consequence of drawing")
        print("the donor pool proportional to positive occurrence. Fix it in")
        print("tcr_swap.py, rebuild, and the baseline should return to 0.5.")
    else:
        print(f"Closest covariate is '{best}', still {best_gap:.4f} away. No single")
        print("receptor-level covariate accounts for it.")
        print()
        print("Stop here. Five hypotheses tested and rejected is enough for one")
        print("effect. Report it as measured:")
        print()
        print("  - a receptor-only model on swap negatives is anti-predictive")
        print("    within epitopes, ~0.45 macro against a simulated null of 0.50")
        print("  - it reproduces across two unrelated model classes and 3 seeds")
        print("  - it is not length, publicness, pool weight, home epitope size,")
        print("    or evaluation structure")
        print("  - so quote every macro against the tcr_only condition, never")
        print("    against 0.5")
        print()
        print("The contribution metric (full minus tcr_only) is unaffected, so")
        print("this does not block anything. It is an open question worth one")
        print("paragraph, not another week.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()