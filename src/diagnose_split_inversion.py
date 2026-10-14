"""Is the random split a valid seen-epitope control? (It is not.)

Three results from diagnose_swap_bias.py set this up.

  partition floor:  pooled 0.5000, and EVERY epitope exactly 0.5000, on three
                    different test sets. That is constant output, not a null
                    result. The instrument collapsed and told us nothing.

  random floor:     0.3164 / 0.2791 / 0.3603 macro, 88-96% of epitopes below
                    chance. Strongly ANTI-predictive from the CDR3 string alone.

  MLP tcr_only:     0.4783 / 0.4770 / 0.4660 macro on the same random split.
                    Same sign, much weaker -- the MLP picks up some of whatever
                    the n-gram model is feasting on, but not most of it.

THE MECHANISM (checked in [2])

Partition-local swap pins each receptor's label total near 50/50: 95.1% of
repeated train receptors are exactly balanced. The median receptor appears about
2.1 times in 300k rows. Split those rows at random and a receptor's train and
test label rates become anti-correlated by sampling WITHOUT REPLACEMENT -- for a
receptor with one positive row and one negative row, if train takes the positive
then test is left holding the negative. Guaranteed, not probabilistic.

This is NOT the §4.5 inversion. That one came from the global donor pool keeping
a receptor in train as a negative while its positive pairing was held out to
test. This one needs no donor pool at all: fixed per-receptor label totals plus a
random row split are sufficient. Same symptom, different cause, and it survives
the partition-local construction that fixed §4.5.

WHY IT MATTERS

§9.5 quotes 0.6878 macro as the seen-epitope ceiling, measured on this random
split. If the receptor channel is anti-predictive there, that number was measured
against a headwind and the real ceiling is higher. The benchmark framing rests on
it, so it needs a split that does not have this problem.

THE PROPOSED FIX (built and measured in [3])

Group rows by receptor and assign whole groups to partitions. Epitopes still
appear in train and test -- the seen-epitope property is preserved -- but no
receptor crosses the boundary, so there is no per-receptor label total to split
and nothing to anti-correlate.

That also gives the 2x2 the project has been missing:

                      receptor seen          receptor unseen
    epitope seen      random split           THIS (clean ceiling)
                      (inverted, unusable)
    epitope unseen    --                     partition split (the floor)

Seen-epitope / unseen-receptor is exactly the axis Liao et al. report models
handle fine, so this control also tests their claim directly on our data.

WHAT IT RUNS

  [1] instrument check -- refit the partition floor and report whether its
                          output is actually constant (distinct scores, score
                          spread, largest coefficient). Settles whether the
                          0.5000 was a collapse or a measurement bug.
  [2] inversion        -- receptor-level train vs test label-rate correlation
                          under the plain random split, in the same units §4.5
                          uses, plus the all-negative-in-train flip rate.
  [3] grouped split    -- build it, verify the seen-epitope property survived,
                          then measure the floor and the same label-rate
                          correlation on it.

READING [3]

  floor macro near 0.5 with NON-constant scores  -> the fix works, the grouped
      split is a clean ceiling control, and the GPU runs are worth doing.
  floor still inverted                           -> grouping did not fix it and
      something else generates the anti-correlation; do not spend GPU time yet.
  floor constant again                           -> instrument dead here too,
      and the ceiling needs a different measurement entirely.

USAGE

    export PYTHONPATH=src
    python3 src/diagnose_split_inversion.py --out results/split_inversion.json
    python3 src/diagnose_split_inversion.py --pkls data/swap_seed0.pkl   # one seed

CPU only, no GPU. About four minutes per seed, dominated by the two floor fits.
Each pkl is ~1.24 GB; embedding columns are dropped on load, one seed at a time.
"""

import argparse
import gc
import json
import os
import re
import statistics as st

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SIZE_BINS = [0, 5, 10, 25, 100, 10 ** 9]
SIZE_LABELS = ["1-5", "6-10", "11-25", "26-100", "101+"]

# Below this many distinct scores the floor model is not measuring anything.
CONSTANT_TOL = 1e-9


def load_slim(path):
    """Read a swap pkl, keep the four columns this needs, drop the rest."""
    df = pd.read_pickle(path)
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("epi", "epitope"):
            rename[c] = "epi"
        if lc in ("tcr", "cdr3"):
            rename[c] = "tcr"
        if lc in ("binding", "label"):
            rename[c] = "binding"
        if lc == "partition":
            rename[c] = "partition"
    df = df.rename(columns=rename)
    missing = {"epi", "tcr", "binding"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    keep = ["epi", "tcr", "binding"] + (
        ["partition"] if "partition" in df.columns else []
    )
    out = df[keep].copy()
    del df
    gc.collect()
    return out


# -- splits --------------------------------------------------------------------

def random_row_split(df, seed, test_frac=0.15, val_frac=0.10):
    """Exactly train_pkl.py --split random."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_test = int(test_frac * len(df))
    n_val = int(val_frac * len(df))
    te = df.iloc[perm[:n_test]].reset_index(drop=True)
    tr = df.iloc[perm[n_test + n_val:]].reset_index(drop=True)
    return tr, te


def receptor_grouped_split(df, seed, test_frac=0.15, val_frac=0.10):
    """Assign whole RECEPTORS to partitions, so none crosses the boundary.

    Same proportional-fill assignment data.py uses for epitopes: each receptor
    goes to whichever partition is currently furthest below its row target.
    Greedy quota-filling would hand the whole test budget to a few very frequent
    receptors, which is the failure mode documented in data.py.

    Receptor frequencies are far less skewed than epitope frequencies here
    (~2.1 rows each), so this is much better behaved than the epitope version --
    but the same code path keeps the two comparable.
    """
    rng = np.random.default_rng(seed)
    counts = df["tcr"].value_counts()

    # Own the strings and sort before permuting, so the split is a function of
    # the receptor SET and the seed only -- see data.py note 3.
    receptors = np.array(sorted(str(r) for r in counts.index), dtype=object)
    sizes = {str(r): int(c) for r, c in counts.items()}
    receptors = rng.permutation(receptors)

    total = len(df)
    targets = {
        "test": total * test_frac,
        "val": total * val_frac,
        "train": total * (1.0 - test_frac - val_frac),
    }
    assigned = {k: [] for k in targets}
    filled = {k: 0 for k in targets}

    for r in receptors:
        part = min(targets, key=lambda k: filled[k] / targets[k])
        assigned[part].append(r)
        filled[part] += sizes[r]

    tr = df[df["tcr"].isin(set(assigned["train"]))].reset_index(drop=True)
    te = df[df["tcr"].isin(set(assigned["test"]))].reset_index(drop=True)

    assert not (set(tr["tcr"]) & set(te["tcr"])), "receptor leaked across split"
    return tr, te


# -- measurements --------------------------------------------------------------

def fit_floor(tr, te):
    """Char n-gram LR on the CDR3 string alone. Returns scores and health info."""
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=3)
    xtr = vec.fit_transform(tr["tcr"])
    xte = vec.transform(te["tcr"])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(xtr, tr["binding"])
    scores = clf.predict_proba(xte)[:, 1]

    uniq = np.unique(scores)
    health = {
        "n_features": int(xtr.shape[1]),
        "n_distinct_scores": int(len(uniq)),
        "score_spread": float(uniq.max() - uniq.min()) if len(uniq) > 1 else 0.0,
        "max_abs_coef": float(np.abs(clf.coef_).max()),
        "mean_abs_coef": float(np.abs(clf.coef_).mean()),
        "collapsed": bool(len(uniq) <= 1
                          or (uniq.max() - uniq.min()) < CONSTANT_TOL),
    }
    return scores, health


def macro_from_scores(te, scores, min_per_class=3):
    y = te["binding"].to_numpy()
    eps = te["epi"].to_numpy()
    aucs, npos = [], []
    for ep in np.unique(eps):
        m = eps == ep
        yl = y[m]
        p, n = int(yl.sum()), int(len(yl) - yl.sum())
        if p < min_per_class or n < min_per_class:
            continue
        aucs.append(float(roc_auc_score(yl, scores[m])))
        npos.append(p)
    if not aucs:
        return {"macro": float("nan"), "n_ep": 0}
    aucs, npos = np.array(aucs), np.array(npos)
    band = pd.cut(npos, SIZE_BINS, labels=SIZE_LABELS)
    by = (pd.DataFrame({"auc": aucs, "band": band})
          .groupby("band", observed=True)["auc"].agg(["mean", "size"]))
    return {
        "macro": round(float(aucs.mean()), 4),
        "n_ep": int(len(aucs)),
        "frac_below_half": round(float((aucs < 0.5).mean()), 4),
        "by_size_band": {str(b): {"macro": round(float(r["mean"]), 4),
                                  "n_ep": int(r["size"])}
                         for b, r in by.iterrows()},
    }


def label_rate_inversion(tr, te):
    """§4.5's statistic: does a receptor's train label rate predict its test one?"""
    a = tr.groupby("tcr")["binding"].mean()
    b = te.groupby("tcr")["binding"].mean()
    shared = a.index.intersection(b.index)
    if len(shared) < 50:
        return {"n_shared_receptors": int(len(shared)), "corr": None}
    av, bv = a.loc[shared].to_numpy(), b.loc[shared].to_numpy()
    corr = (float(np.corrcoef(av, bv)[0, 1])
            if av.std() > 0 and bv.std() > 0 else float("nan"))
    all_neg = av == 0.0
    all_pos = av == 1.0
    return {
        "n_shared_receptors": int(len(shared)),
        "frac_of_test_receptors_shared": round(
            float(len(shared) / te["tcr"].nunique()), 4),
        "corr": round(corr, 4),
        "all_negative_in_train_pct_positive_in_test": (
            round(float(bv[all_neg].mean() * 100), 2) if all_neg.any() else None),
        "all_positive_in_train_pct_positive_in_test": (
            round(float(bv[all_pos].mean() * 100), 2) if all_pos.any() else None),
    }


# -- main ----------------------------------------------------------------------

def seed_from_path(path, fallback=0):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else fallback


def show_floor(tag, health, macro, pooled):
    flag = "  <-- COLLAPSED, no reading" if health["collapsed"] else ""
    print(f"      {tag}: pooled {pooled:.4f}   macro {macro['macro']:.4f}"
          f" over {macro['n_ep']} epitopes{flag}")
    print(f"        distinct scores {health['n_distinct_scores']:,}"
          f"   spread {health['score_spread']:.3e}"
          f"   max|coef| {health['max_abs_coef']:.4f}")
    if not health["collapsed"] and macro["n_ep"]:
        print(f"        below 0.5: {100 * macro['frac_below_half']:.1f}% of epitopes")
        for b in SIZE_LABELS:
            if b in macro["by_size_band"]:
                v = macro["by_size_band"][b]
                print(f"          {b:<8}{v['macro']:>9.4f}  (n={v['n_ep']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkls", nargs="+",
                    default=[f"data/swap_seed{s}.pkl" for s in (0, 1, 2)])
    ap.add_argument("--min-per-class", type=int, default=3,
                    help="must match train_pkl.py's macro_auroc filter")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}

    for path in args.pkls:
        if not os.path.exists(path):
            print(f"!! missing {path}, skipping")
            continue
        seed = seed_from_path(path)
        print(f"\n{'=' * 76}\n{path}   (seed {seed})\n{'=' * 76}")
        df = load_slim(path)
        occ = df["tcr"].value_counts()
        print(f"  {len(df):,} rows, {df.epi.nunique()} epitopes, "
              f"{len(occ):,} distinct receptors "
              f"(mean {len(df) / len(occ):.2f} rows each, median {int(occ.median())})")
        r = results.setdefault(f"seed{seed}", {"seed": seed})

        # ---- [1] is the partition floor actually constant? -------------------
        print("\n  [1] INSTRUMENT CHECK -- partition floor")
        if "partition" in df.columns:
            tr = df[df.partition == "train"].reset_index(drop=True)
            te = df[df.partition == "test"].reset_index(drop=True)
            s, h = fit_floor(tr, te)
            m = macro_from_scores(te, s, args.min_per_class)
            pooled = float(roc_auc_score(te["binding"], s))
            r["partition_floor"] = {"health": h, "macro": m,
                                    "pooled": round(pooled, 4)}
            show_floor("partition", h, m, pooled)
            del tr, te, s
        else:
            print("      no partition column")

        # ---- [2] the inversion, plain random split ---------------------------
        print("\n  [2] RECEPTOR LABEL-RATE INVERSION -- plain random split")
        tr, te = random_row_split(df, seed)
        inv = label_rate_inversion(tr, te)
        r["random_inversion"] = inv
        print(f"      receptors in both train and test: "
              f"{inv['n_shared_receptors']:,} "
              f"({100 * inv.get('frac_of_test_receptors_shared', 0):.1f}% of test)")
        print(f"      corr(train label rate, test label rate) = {inv['corr']}")
        if inv["all_negative_in_train_pct_positive_in_test"] is not None:
            print(f"      all-negative in train -> "
                  f"{inv['all_negative_in_train_pct_positive_in_test']:.1f}% positive in test")
        if inv["all_positive_in_train_pct_positive_in_test"] is not None:
            print(f"      all-positive in train -> "
                  f"{inv['all_positive_in_train_pct_positive_in_test']:.1f}% positive in test")
        s, h = fit_floor(tr, te)
        m = macro_from_scores(te, s, args.min_per_class)
        pooled = float(roc_auc_score(te["binding"], s))
        r["random_floor"] = {"health": h, "macro": m, "pooled": round(pooled, 4)}
        show_floor("random", h, m, pooled)
        del tr, te, s

        # ---- [3] the proposed fix -------------------------------------------
        print("\n  [3] RECEPTOR-GROUPED SPLIT (seen epitopes, unseen receptors)")
        tr, te = receptor_grouped_split(df, seed)
        shared_ep = len(set(tr.epi) & set(te.epi))
        te_rate = te.groupby("epi")["binding"].mean()
        cons = {
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "epitopes_shared": int(shared_ep),
            "test_epitopes": int(te.epi.nunique()),
            "frac_test_epitopes_seen": round(
                float(shared_ep / max(te.epi.nunique(), 1)), 4),
            "receptor_overlap": 0,
            "test_positive_rate": round(float(te.binding.mean()), 4),
            "test_epitopes_off_50pct_by_5pp": int((te_rate.sub(0.5).abs() > 0.05).sum()),
        }
        r["grouped_construction"] = cons
        print(f"      train {cons['n_train']:,} / test {cons['n_test']:,}, "
              f"receptor overlap 0 (asserted)")
        print(f"      test epitopes {cons['test_epitopes']}, of which "
              f"{shared_ep} seen in train "
              f"({100 * cons['frac_test_epitopes_seen']:.1f}%)")
        print(f"      test positive rate {cons['test_positive_rate']:.4f}, "
              f"{cons['test_epitopes_off_50pct_by_5pp']} epitopes more than "
              f"5pp off 50%")

        inv_g = label_rate_inversion(tr, te)
        r["grouped_inversion"] = inv_g
        print(f"      receptors in both: {inv_g['n_shared_receptors']:,} "
              f"(must be 0)")
        s, h = fit_floor(tr, te)
        m = macro_from_scores(te, s, args.min_per_class)
        pooled = float(roc_auc_score(te["binding"], s))
        r["grouped_floor"] = {"health": h, "macro": m, "pooled": round(pooled, 4)}
        show_floor("grouped", h, m, pooled)

        del tr, te, s, df
        gc.collect()

    # -- verdict ---------------------------------------------------------------
    print(f"\n{'=' * 76}\nVERDICT\n{'=' * 76}")

    def gather(key, field="macro"):
        out = []
        for v in results.values():
            if key in v and not v[key]["health"]["collapsed"]:
                out.append(v[key]["macro"][field])
        return out

    pc = [v["partition_floor"]["health"]["collapsed"]
          for v in results.values() if "partition_floor" in v]
    if pc and all(pc):
        print("\n[1] The partition floor IS constant output. §4.5's caveat holds:")
        print("    the n-gram floor cannot measure this construction, and the")
        print("    0.5000 it prints is the instrument, not the data. The")
        print("    sub-chance tcr_only macro on partition stays OPEN and needs")
        print("    saved predictions, not this model.")
    elif pc:
        print("\n[1] The partition floor did NOT collapse everywhere. A real")
        print("    reading exists on at least one seed -- read the numbers above")
        print("    rather than trusting the earlier 0.5000.")

    corrs = [v["random_inversion"]["corr"] for v in results.values()
             if v.get("random_inversion", {}).get("corr") is not None]
    if corrs:
        print(f"\n[2] Random-split receptor label-rate correlation: "
              f"{st.mean(corrs):+.3f} "
              f"({', '.join(f'{c:+.3f}' for c in corrs)})")
        if st.mean(corrs) < -0.2:
            print("    Negative, as predicted. Fixed per-receptor label totals plus")
            print("    a random row split anti-correlate train and test by sampling")
            print("    without replacement. The random split is NOT a clean")
            print("    seen-epitope control, and the 0.6878 ceiling in §9.5 was")
            print("    measured against this headwind.")
        else:
            print("    Not negative. The inverted random floor comes from something")
            print("    other than per-receptor label totals; chase it before")
            print("    rewriting §9.5.")

    gf, rf = gather("grouped_floor"), gather("random_floor")
    gcol = [v["grouped_floor"]["health"]["collapsed"] for v in results.values()
            if "grouped_floor" in v]
    print()
    if gcol and all(gcol):
        print("[3] The grouped floor collapsed too. No reading; the ceiling needs")
        print("    a different instrument before any GPU time is spent.")
    elif gf:
        print(f"[3] Grouped floor macro {st.mean(gf):.4f} "
              f"+- {st.pstdev(gf):.4f}   ({', '.join(f'{x:.4f}' for x in gf)})")
        if rf:
            print(f"    Random floor for comparison: {st.mean(rf):.4f}")
        if abs(st.mean(gf) - 0.5) < 0.03:
            print("    -> The fix works. Grouping removes the inversion while")
            print("       keeping epitopes shared, so this is a clean")
            print("       seen-epitope ceiling. Worth the GPU runs: full and")
            print("       tcr_only, 3 seeds, as a third --split mode.")
        else:
            print("    -> Still off chance. Grouping did not remove it; find the")
            print("       remaining channel before spending GPU time.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()