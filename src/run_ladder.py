"""Run the shortcut-removal ladder and report the provenance floor at each rung.

Interventions are applied CUMULATIVELY: rung 3 has also had rungs 1 and 2
applied. After each one, the provenance floor is remeasured -- a character
n-gram logistic regression on the CDR3 string alone, no peptide, on an
epitope-disjoint split.

The floor is the right metric here because it is architecture-free. It measures
what the DATA makes available, not what one model happens to extract, so a drop
between rungs is attributable to the intervention rather than to training noise.

    python3 src/run_ladder.py --csv data/TCREpitopePairs.csv --seeds 0 1 2

Roughly two minutes per rung per seed on CPU. No GPU needed.
"""

import argparse
import json
import os
import statistics as st

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data import epitope_disjoint_split, load_pairs, subsample
from interventions import LADDER, is_malformed


def fixed_partition(df, seed):
    """Assign epitopes to partitions ONCE, on the baseline data.

    Every intervention changes the row set, which changes epitope sizes, which
    changes how the proportional-fill splitter assigns epitopes. Re-splitting
    at each rung therefore measures the intervention PLUS a fresh split, and
    the split term is large -- test epitope counts swung between 100 and 249
    across rungs in the first version of this script, and only the largest
    intervention produced an effect bigger than that noise.

    So the partition is computed once here and reused for every rung. Each
    rung then filters its own rows into that fixed assignment, and any change
    in the floor is attributable to the intervention.
    """
    train, val, test = epitope_disjoint_split(df, seed=seed)
    return {
        "train": set(train["epitope"]),
        "val": set(val["epitope"]),
        "test": set(test["epitope"]),
    }


def provenance_floor(df, partition):
    """AUROC reachable from the CDR3 string alone, on a FIXED partition."""
    train = df[df["epitope"].isin(partition["train"])]
    test = df[df["epitope"].isin(partition["test"])]
    if len(train) < 100 or len(test) < 100:
        return None

    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=3)
    xtr = vec.fit_transform(train["cdr3"])
    xte = vec.transform(test["cdr3"])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(xtr, train["label"])
    auc = roc_auc_score(test["label"], clf.predict_proba(xte)[:, 1])

    # Length alone, for comparison: how much of the floor is just length?
    lens = test["cdr3"].str.len().to_numpy()
    auc_len = roc_auc_score(test["label"], -lens)

    return {
        "floor": round(float(auc), 4),
        "length_only": round(float(auc_len), 4),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "test_epitopes": int(test["epitope"].nunique()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-rows", type=int, default=300016)
    ap.add_argument("--out", default="results/ladder.json")
    args = ap.parse_args()

    base = subsample(load_pairs(args.csv), args.max_rows, seed=0)
    print(f"loaded {len(base):,} pairs, "
          f"{100 * is_malformed(base['cdr3']).mean():.2f}% malformed\n")

    results = {}
    for seed in args.seeds:
        print(f"{'=' * 62}\nSEED {seed}\n{'=' * 62}")
        part = fixed_partition(base, seed)
        print(f"  partition fixed: {len(part['train'])} train / "
              f"{len(part['val'])} val / {len(part['test'])} test epitopes")
        df = base
        for name, fn in LADDER:
            df = fn(df, seed)
            r = provenance_floor(df, part)
            if r is None:
                print(f"  {name:<18} SKIPPED (too few rows left)")
                continue
            results.setdefault(name, []).append(r)
            print(f"  {name:<18} floor {r['floor']:.4f}   "
                  f"len-only {r['length_only']:.4f}   "
                  f"rows {r['n_train'] + r['n_test']:>7,}   "
                  f"test eps {r['test_epitopes']}")

        # Size control: random downsample to the final rung's size, so the
        # effect of "less data" can be separated from the effect of matching.
        final_n = len(df)
        ctrl = base.sample(n=min(final_n, len(base)), random_state=seed)
        rc = provenance_floor(ctrl, part)
        if rc:
            results.setdefault("size_control", []).append(rc)
            print(f"  {'size_control':<18} floor {rc['floor']:.4f}   "
                  f"len-only {rc['length_only']:.4f}   "
                  f"rows {rc['n_train'] + rc['n_test']:>7,}   "
                  f"(random subsample to final rung size)")
        print()

    print("=" * 78)
    print("LADDER SUMMARY -- provenance floor, FIXED partition per seed")
    print("=" * 78)
    print(f"{'rung':<20}{'floor':>9}{'sd':>8}{'len-only':>11}"
          f"{'drop':>9}{'drop/sd':>9}")
    print("-" * 78)
    prev = None
    for name, _ in LADDER:
        if name not in results:
            continue
        fl = [r["floor"] for r in results[name]]
        ln = [r["length_only"] for r in results[name]]
        m, sd = st.mean(fl), st.pstdev(fl)
        if prev is None:
            drop, ratio = "", ""
        else:
            d = m - prev
            drop = f"{d:+.4f}"
            ratio = f"{abs(d) / max(sd, 1e-9):.1f}"
        print(f"{name:<20}{m:>9.4f}{sd:>8.4f}{st.mean(ln):>11.4f}"
              f"{drop:>9}{ratio:>9}")
        prev = m
    print("-" * 78)
    if "size_control" in results:
        cf = [r["floor"] for r in results["size_control"]]
        print(f"{'size_control':<20}{st.mean(cf):>9.4f}{st.pstdev(cf):>8.4f}"
              f"{'':>11}{'':>9}{'':>9}")
        print("  (baseline data randomly downsampled to the final rung's size)")
    print("-" * 78)
    print("\nA rung's drop is only believable if drop/sd is comfortably above 1.")
    print("Compare the final rung against size_control, not against baseline:")
    print("the difference between them is what MATCHING bought, as opposed to")
    print("what simply using less data did.")

    first = st.mean([r["floor"] for r in results[LADDER[0][0]]])
    last = st.mean([r["floor"] for r in results[LADDER[-1][0]]])
    print(f"\ntotal floor reduction: {first:.4f} -> {last:.4f} "
          f"({first - last:+.4f})")
    print(f"distance still above chance: {last - 0.5:+.4f}")
    print()
    if last > 0.60:
        print("The floor stays well above chance after every intervention that")
        print("can be applied to the sequences. That points at the negatives")
        print("themselves rather than at anything fixable by cleaning, and")
        print("motivates rebuilding them by TCR swap.")
    else:
        print("The floor approaches chance, so the shortcut is largely an")
        print("artifact of sequence formatting and composition and the dataset")
        print("can be repaired without rebuilding the negatives.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()