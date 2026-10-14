"""Diagnose why epi_only macro AUROC is 0.5624 instead of exactly 0.5000.

In epi_only the TCR vector is zeroed, so within one epitope EVERY row has an
identical input: same epitope embedding, same zero vector. The model is a
deterministic function, so it must emit one identical score for all of them.
AUROC over identical scores is exactly 0.5. Our CNN run gave 0.5000. The
BLOSUM62/MLP run gave 0.5624, so something is breaking the ties.

This is a NUMERICS question, not a weights question -- "do identical inputs
produce identical outputs" does not depend on training -- so this runs on a
freshly initialised model in seconds, no 26-minute retrain needed.

    python3 src/diagnose_epi_only.py --pkl data/BLOSUM62.pkl

Checks, in order:
  1. Row ordering in the pkl (are positives grouped before negatives?)
  2. Are epitope embeddings actually identical within an epitope?
  3. Does inference emit identical scores within an epitope?
  4. If not, how large are the differences, and do they track the label?
  5. What macro AUROC do those differences produce?
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data import epitope_disjoint_split
from train_pkl import BindingMLP, apply_ablation, load_pkl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=4096)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = load_pkl(args.pkl)
    _, _, test_df = epitope_disjoint_split(df, seed=args.seed)
    print(f"test set: {len(test_df):,} rows, "
          f"{test_df.epitope.nunique()} epitopes, device={device}\n")

    # ---- 1. row ordering -----------------------------------------------------
    lab = test_df["label"].to_numpy()
    flips = int((np.diff(lab) != 0).sum())
    print("[1] ROW ORDERING")
    print(f"    label sequence changes value {flips:,} times "
          f"in {len(lab):,} rows")
    if flips < 50:
        print("    -> labels are strongly GROUPED (positives then negatives).")
        print("       Within an epitope, positives and negatives land in")
        print("       different eval batches.")
    else:
        print("    -> labels are interleaved; grouping is not the issue.")

    # ---- 2. are inputs identical within an epitope? --------------------------
    print("\n[2] ARE EPITOPE EMBEDDINGS IDENTICAL WITHIN AN EPITOPE?")
    bad = 0
    for ep, g in list(test_df.groupby("epitope"))[:20]:
        arr = np.stack(g["epi_embeds"].to_numpy())
        if not np.all(arr == arr[0]):
            bad += 1
    print(f"    checked 20 epitopes, {bad} had non-identical embeddings")
    if bad:
        print("    -> the inputs themselves differ; that alone explains it.")
    else:
        print("    -> inputs are identical, so outputs must be too.")

    # ---- 3. run inference ----------------------------------------------------
    print("\n[3] RUNNING epi_only INFERENCE (untrained model, eval mode)")
    tcr = torch.from_numpy(
        np.asarray(np.stack(test_df["tcr_embeds"].to_numpy()), dtype=np.float32)
    ).to(device)
    epi = torch.from_numpy(
        np.asarray(np.stack(test_df["epi_embeds"].to_numpy()), dtype=np.float32)
    ).to(device)

    model = BindingMLP(tcr.shape[1], epi.shape[1]).to(device).eval()
    scores = np.empty(len(test_df), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(test_df), args.eval_batch_size):
            t, e = apply_ablation(
                tcr[i:i + args.eval_batch_size],
                epi[i:i + args.eval_batch_size],
                "epi_only",
            )
            scores[i:i + args.eval_batch_size] = (
                torch.sigmoid(model(t, e)).double().cpu().numpy()
            )

    # ---- 4. distinct scores per epitope -------------------------------------
    print("\n[4] DISTINCT SCORES PER EPITOPE")
    eps = test_df["epitope"].to_numpy()
    n_distinct, spreads = [], []
    for ep in np.unique(eps):
        m = eps == ep
        u = np.unique(scores[m])
        n_distinct.append(len(u))
        if len(u) > 1:
            spreads.append(u.max() - u.min())

    exact = sum(1 for n in n_distinct if n == 1)
    print(f"    epitopes with exactly 1 distinct score: "
          f"{exact} / {len(n_distinct)}")
    if spreads:
        print(f"    among the rest, score spread: "
              f"min {min(spreads):.3e}, median {np.median(spreads):.3e}, "
              f"max {max(spreads):.3e}")
        if max(spreads) < 1e-4:
            print("    -> differences are FLOATING-POINT NOISE, not signal.")
        else:
            print("    -> differences are too large to be float noise.")

    # ---- 5. what macro AUROC does that produce? ------------------------------
    print("\n[5] RESULTING MACRO AUROC")
    aucs, degenerate = [], 0
    for ep in np.unique(eps):
        m = eps == ep
        yl = lab[m]
        if yl.sum() < 3 or (len(yl) - yl.sum()) < 3:
            continue
        a = roc_auc_score(yl, scores[m])
        aucs.append(a)
        if a in (0.0, 1.0):
            degenerate += 1
    if aucs:
        print(f"    macro AUROC over {len(aucs)} epitopes: {np.mean(aucs):.4f}")
        print(f"    epitopes scoring exactly 0.0 or 1.0: {degenerate}")
        print(f"    epitopes scoring exactly 0.5:        "
              f"{sum(1 for a in aucs if a == 0.5)}")

    print("\n" + "=" * 68)
    if spreads and max(spreads) < 1e-4 and degenerate > 0:
        print("CONCLUSION: floating-point tie-breaking.")
        print("Identical inputs give scores differing ~1e-7 depending on")
        print("batch composition. Because the rows are label-ordered, those")
        print("differences correlate with the label, so per-epitope AUROC")
        print("collapses to 0.0 or 1.0 instead of 0.5. The macro number is a")
        print("measurement artifact and should be reported as 0.5.")
        print()
        print("It does NOT affect the headline result: the epitope")
        print("contribution comes from full minus tcr_only.")
    elif exact == len(n_distinct):
        print("CONCLUSION: scores ARE constant per epitope here.")
        print("The 0.5624 then came from the trained model, not the")
        print("architecture -- rerun with --save-preds to confirm.")
    else:
        print("CONCLUSION: inconclusive; see numbers above.")
    print("=" * 68)


if __name__ == "__main__":
    main()