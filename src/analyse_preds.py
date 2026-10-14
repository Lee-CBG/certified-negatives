"""Analyse raw test predictions saved by train_pkl.py --save-preds.

This inspects the ACTUAL TRAINED MODEL's outputs, so it settles the question
the untrained diagnostic can only approximate.

    python3 src/analyse_preds.py --npz results/preds_epi_only.npz

For epi_only, every row sharing an epitope had an identical input, so every
score within an epitope should be identical and every per-epitope AUROC should
be exactly 0.5.
"""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    scores, labels, eps = d["scores"], d["labels"], d["epitopes"]
    print(f"{len(scores):,} predictions over {len(np.unique(eps))} epitopes\n")

    print("[1] DISTINCT SCORES PER EPITOPE")
    n_distinct, spreads = [], []
    for ep in np.unique(eps):
        m = eps == ep
        u = np.unique(scores[m])
        n_distinct.append(len(u))
        if len(u) > 1:
            spreads.append(float(u.max() - u.min()))

    exact = sum(1 for n in n_distinct if n == 1)
    print(f"    exactly 1 distinct score: {exact} / {len(n_distinct)} epitopes")
    print(f"    median distinct values:   {int(np.median(n_distinct))}")
    if spreads:
        print(f"    score spread among the rest:")
        print(f"      min    {min(spreads):.3e}")
        print(f"      median {np.median(spreads):.3e}")
        print(f"      max    {max(spreads):.3e}")

    print("\n[2] PER-EPITOPE AUROC DISTRIBUTION")
    aucs = []
    for ep in np.unique(eps):
        m = eps == ep
        yl = labels[m]
        if yl.sum() < 3 or (len(yl) - yl.sum()) < 3:
            continue
        aucs.append(roc_auc_score(yl, scores[m]))
    aucs = np.array(aucs)
    if len(aucs):
        print(f"    epitopes scored: {len(aucs)}")
        print(f"    macro AUROC:     {aucs.mean():.4f}")
        print(f"    exactly 0.5:     {(aucs == 0.5).sum()}")
        print(f"    exactly 0.0:     {(aucs == 0.0).sum()}")
        print(f"    exactly 1.0:     {(aucs == 1.0).sum()}")
        print(f"    strictly between: {((aucs > 0) & (aucs < 1) & (aucs != 0.5)).sum()}")

    print("\n[3] DO SCORE DIFFERENCES TRACK THE LABEL?")
    tracked = 0
    checked = 0
    for ep in np.unique(eps):
        m = eps == ep
        yl, sl = labels[m], scores[m]
        if yl.sum() < 3 or (len(yl) - yl.sum()) < 3:
            continue
        checked += 1
        pos_mean = sl[yl == 1].mean()
        neg_mean = sl[yl == 0].mean()
        if abs(pos_mean - neg_mean) > 0:
            tracked += 1
    print(f"    epitopes where pos/neg means differ at all: {tracked}/{checked}")

    print("\n" + "=" * 68)
    if exact == len(n_distinct):
        print("Scores are constant within every epitope, as they must be.")
        print("Macro AUROC should be exactly 0.5. If the training run")
        print("reported otherwise, the bug is in macro_auroc, not the model.")
    elif spreads and max(spreads) < 1e-4:
        print("CONFIRMED: floating-point tie-breaking.")
        print(f"Differences are ~{np.median(spreads):.1e} -- far below any real")
        print("signal, but enough to break ties. Because rows are label-")
        print("ordered, ties break in a label-correlated direction and")
        print("per-epitope AUROC collapses to 0.0 or 1.0 rather than 0.5.")
        print()
        print("Report epi_only macro AUROC as 0.5. The headline result is")
        print("unaffected: it comes from full minus tcr_only.")
    else:
        print("Differences are larger than float noise -- investigate further.")
    print("=" * 68)


if __name__ == "__main__":
    main()