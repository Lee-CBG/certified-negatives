"""Interpret the provenance-floor model: what is the shortcut made of?

The floor model is a logistic regression on character n-grams of the CDR3
string, so its coefficients are directly readable. A positive coefficient
means that substring pushes a sequence toward the "binding" label -- which,
given how the negatives were built, really means "toward the positive FILE".

Three caveats on reading these numbers:

  1. Features are TF-IDF weighted, not raw counts, so a coefficient is per
     unit of TF-IDF, not per occurrence.
  2. N-grams are heavily nested and correlated (S, SS, SSL all fire together),
     so L2 regularisation splits credit among them. Treat groups of related
     n-grams as one signal, not several.
  3. This shows what the model USES, not what causes anything. The whole point
     is that these features are artifacts of data provenance.

    python3 src/interpret_ngram.py --csv data/TCREpitopePairs.csv \
        --max-rows 300016 --seed 0
"""

import argparse
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data import epitope_disjoint_split, load_pairs, subsample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-rows", type=int, default=300016)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--export", default=None,
                    help="write the full ranked coefficient table to this path")
    args = ap.parse_args()

    df = subsample(load_pairs(args.csv), args.max_rows, seed=args.seed)
    train, _, test = epitope_disjoint_split(df, seed=args.seed)
    print(f"train {len(train):,} / test {len(test):,}\n")

    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=3)
    xtr = vec.fit_transform(train["cdr3"])
    xte = vec.transform(test["cdr3"])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(xtr, train["label"])
    auc = roc_auc_score(test["label"], clf.predict_proba(xte)[:, 1])
    print(f"AUROC {auc:.4f}   features {xtr.shape[1]:,}\n")

    names = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]
    order = np.argsort(coef)

    print("=" * 62)
    print(f"{'PUSHES TOWARD POSITIVE':<31}{'PUSHES TOWARD NEGATIVE':<31}")
    print("=" * 62)
    for i in range(args.top):
        p = order[-(i + 1)]
        n = order[i]
        print(f"  {names[p]:<8} {coef[p]:+.3f}"
              f"{'':<12}  {names[n]:<8} {coef[n]:+.3f}")

    # Single amino acids only: directly comparable to the frequency deltas
    print("\n" + "=" * 62)
    print("SINGLE AMINO ACIDS ONLY (comparable to the audit's freq deltas)")
    print("=" * 62)
    singles = [(n, c) for n, c in zip(names, coef) if len(n) == 1]
    singles.sort(key=lambda t: -t[1])
    for n, c in singles[:8]:
        print(f"  {n}  {c:+.3f}")
    print("  ...")
    for n, c in singles[-8:]:
        print(f"  {n}  {c:+.3f}")

    # How much signal sits at each n-gram length?
    print("\n" + "=" * 62)
    print("WHERE THE SIGNAL LIVES, BY N-GRAM LENGTH")
    print("=" * 62)
    by_len = defaultdict(list)
    for n, c in zip(names, coef):
        by_len[len(n)].append(abs(c))
    for L in sorted(by_len):
        v = np.array(by_len[L])
        print(f"  length {L}: {len(v):>6,} features   "
              f"mean |coef| {v.mean():.4f}   max {v.max():.3f}")

    # Length effect: does the model just count residues?
    print("\n" + "=" * 62)
    print("IS IT JUST SEQUENCE LENGTH?")
    print("=" * 62)
    scores = clf.decision_function(xte)
    lens = test["cdr3"].str.len().to_numpy()
    r = np.corrcoef(scores, lens)[0, 1]
    print(f"  corr(model score, CDR3 length) = {r:+.3f}")
    auc_len = roc_auc_score(test["label"], -lens)
    print(f"  AUROC using NEGATIVE length alone = {auc_len:.4f}")
    print("  (positives are shorter, so shorter -> more positive)")

    if args.export:
        pos = sorted([(n, c) for n, c in zip(names, coef) if c > 0],
                     key=lambda t: -t[1])
        neg = sorted([(n, c) for n, c in zip(names, coef) if c < 0],
                     key=lambda t: t[1])
        with open(args.export, "w") as f:
            f.write("RANKED COEFFICIENTS OF THE PROVENANCE-FLOOR MODEL\n")
            f.write("=" * 66 + "\n\n")
            f.write("Model: logistic regression (C=1.0, L2) on character n-grams\n")
            f.write("       of the CDR3 string, ngram_range=(1,4), TF-IDF weighted,\n")
            f.write("       min_df=3. The peptide was never shown to this model.\n\n")
            f.write(f"Data:      {args.csv}\n")
            f.write(f"Split:     epitope-disjoint, seed {args.seed}\n")
            f.write(f"Train:     {len(train):,} pairs\n")
            f.write(f"Test:      {len(test):,} pairs, "
                    f"{test.epitope.nunique()} unseen epitopes\n")
            f.write(f"Features:  {len(names):,}\n")
            f.write(f"Test AUROC: {auc:.4f}\n\n")
            f.write("A POSITIVE coefficient pushes a sequence toward the label\n")
            f.write("'binding'. Because the negatives were drawn from a separate\n")
            f.write("background repertoire, that really means 'toward the positive\n")
            f.write("source file' rather than toward any binding property.\n\n")
            f.write("READ WITH CARE:\n")
            f.write("  - Features are TF-IDF weighted, so a coefficient is per unit\n")
            f.write("    of TF-IDF, not per occurrence of the substring.\n")
            f.write("  - N-grams are nested and correlated (S, SS, SSL all fire\n")
            f.write("    together), so L2 splits credit among them. Read groups of\n")
            f.write("    related n-grams as one signal, not as separate findings.\n")
            f.write("  - Single-feature weights are unstable across seeds even when\n")
            f.write("    the grouped signal is not. Refit across several seeds and\n")
            f.write("    keep only what survives before quoting anything.\n")
            f.write("  - These describe what the model USES, not what causes\n")
            f.write("    binding. The point is that they are provenance artifacts.\n\n")

            f.write("=" * 66 + "\n")
            f.write(f"SECTION 1: PUSHES TOWARD POSITIVE  ({len(pos):,} features)\n")
            f.write("=" * 66 + "\n")
            f.write(f"{'rank':>7}  {'n-gram':<10}{'len':>4}  {'coefficient':>12}\n")
            f.write("-" * 66 + "\n")
            for i, (n, c) in enumerate(pos, 1):
                f.write(f"{i:>7}  {n:<10}{len(n):>4}  {c:>+12.5f}\n")

            f.write("\n" + "=" * 66 + "\n")
            f.write(f"SECTION 2: PUSHES TOWARD NEGATIVE  ({len(neg):,} features)\n")
            f.write("=" * 66 + "\n")
            f.write(f"{'rank':>7}  {'n-gram':<10}{'len':>4}  {'coefficient':>12}\n")
            f.write("-" * 66 + "\n")
            for i, (n, c) in enumerate(neg, 1):
                f.write(f"{i:>7}  {n:<10}{len(n):>4}  {c:>+12.5f}\n")

            f.write("\n" + "=" * 66 + "\n")
            f.write("SECTION 3: SINGLE AMINO ACIDS, RANKED\n")
            f.write("=" * 66 + "\n")
            f.write(f"{'residue':>9}  {'coefficient':>12}\n")
            f.write("-" * 66 + "\n")
            for n, c in singles:
                f.write(f"{n:>9}  {c:>+12.5f}\n")
        print(f"\nFull ranked table written to {args.export}")


if __name__ == "__main__":
    main()