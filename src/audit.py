"""STEP 1 -- Audit the training data before training anything.

Four checks, each of which can independently invalidate the whole experiment:

  1. CDR3 overlap between classes. If positive and negative CDR3s are disjoint
     sets, the label is partly a property of WHICH FILE a sequence came from,
     not of whether it binds anything.

  2. Per-epitope class balance. If every epitope sits at exactly 50% positive,
     the epitope carries no marginal information about the label.

  3. Provenance leakage, quantified. A character n-gram logistic regression on
     the CDR3 STRING ALONE -- never shown a peptide. Whatever AUROC this gets
     is achievable without modelling binding at all. It is the floor any real
     model has to beat to be doing anything.

  4. Surface statistics: length and amino acid frequency by class. These show
     WHAT the n-gram model is picking up on.

Run this first. If check 3 comes back high, stop and read the output before
training.
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from blosum import AA
from data import load_pairs


def check_cdr3_overlap(df):
    pos = set(df.loc[df.label == 1, "cdr3"])
    neg = set(df.loc[df.label == 0, "cdr3"])
    shared = pos & neg
    return {
        "unique_positive_cdr3": len(pos),
        "unique_negative_cdr3": len(neg),
        "shared_cdr3": len(shared),
        "shared_pct_of_positive": round(100 * len(shared) / max(len(pos), 1), 3),
    }


def check_epitope_balance(df):
    g = df.groupby("epitope")["label"].agg(["mean", "size"])
    return {
        "n_epitopes": int(len(g)),
        "median_positive_rate": round(float(g["mean"].median()), 4),
        "mean_positive_rate": round(float(g["mean"].mean()), 4),
        "pct_epitopes_at_exactly_50pct": round(
            100 * float((g["mean"].round(6) == 0.5).mean()), 2
        ),
        "median_pairs_per_epitope": int(g["size"].median()),
    }


def check_provenance_leakage(df, seed=0, max_rows=120_000):
    """Train a logistic regression on CDR3 character n-grams only.

    IMPORTANT: split by epitope here too, so this number is comparable to the
    CNN's epitope-disjoint test AUROC.
    """
    from data import epitope_disjoint_split, subsample

    sub = subsample(df, max_rows, seed=seed)
    train, _, test = epitope_disjoint_split(sub, seed=seed)

    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=3)
    xtr = vec.fit_transform(train["cdr3"])
    xte = vec.transform(test["cdr3"])

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(xtr, train["label"])
    auc = roc_auc_score(test["label"], clf.predict_proba(xte)[:, 1])

    return {
        "cdr3_only_ngram_auroc": round(float(auc), 4),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "note": "No peptide was shown to this model. This is the provenance floor.",
    }


def check_surface_stats(df):
    pos = df[df.label == 1]["cdr3"]
    neg = df[df.label == 0]["cdr3"]

    def aa_freq(seqs):
        c = Counter()
        total = 0
        for s in seqs:
            c.update(s)
            total += len(s)
        return {a: 100 * c.get(a, 0) / total for a in AA}

    fp, fn = aa_freq(pos), aa_freq(neg)
    deltas = sorted(
        ((a, round(fp[a] - fn[a], 3)) for a in AA),
        key=lambda t: -abs(t[1]),
    )
    return {
        "mean_len_positive": round(float(pos.str.len().mean()), 3),
        "mean_len_negative": round(float(neg.str.len().mean()), 3),
        "mean_len_difference": round(
            float(pos.str.len().mean() - neg.str.len().mean()), 3
        ),
        "largest_aa_freq_deltas_pct_points": deltas[:6],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results/audit.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=120_000)
    args = ap.parse_args()

    df = load_pairs(args.csv)
    print(f"Loaded {len(df):,} pairs, {df.label.mean():.4f} positive rate\n")

    report = {
        "n_pairs": int(len(df)),
        "overall_positive_rate": round(float(df.label.mean()), 4),
        "cdr3_overlap": check_cdr3_overlap(df),
        "epitope_balance": check_epitope_balance(df),
        "surface_stats": check_surface_stats(df),
    }

    print("[1/4] CDR3 overlap between classes")
    for k, v in report["cdr3_overlap"].items():
        print(f"      {k:32s} {v:,}" if isinstance(v, int) else f"      {k:32s} {v}")

    print("\n[2/4] Per-epitope class balance")
    for k, v in report["epitope_balance"].items():
        print(f"      {k:32s} {v}")

    print("\n[3/4] Surface statistics by class")
    for k, v in report["surface_stats"].items():
        print(f"      {k:32s} {v}")

    print("\n[4/4] Provenance leakage (this one takes a minute)...")
    report["provenance_leakage"] = check_provenance_leakage(
        df, seed=args.seed, max_rows=args.max_rows
    )
    for k, v in report["provenance_leakage"].items():
        print(f"      {k:32s} {v}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    auc = report["provenance_leakage"]["cdr3_only_ngram_auroc"]
    print("\n" + "=" * 68)
    print(f"VERDICT: CDR3 string alone, no peptide, reaches AUROC {auc}")
    if auc > 0.6:
        print("  This is a provenance confound. Any model trained on this data")
        print("  can score well without using the peptide at all. Interpret")
        print("  the ablation results in step 2 with this number in hand.")
    else:
        print("  Low enough that the label is not trivially recoverable from")
        print("  the CDR3 alone. Proceed to step 2.")
    print("=" * 68)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()