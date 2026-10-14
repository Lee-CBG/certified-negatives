"""Collect three ablation runs into one verdict table.

Works for both the CNN runs (default) and the pkl/MLP runs:
    python3 src/report.py                       # CNN results
    python3 src/report.py --prefix pkl_         # BLOSUM62 pkl results
"""

import argparse
import json
import os


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--prefix", default="",
                    help="filename prefix, e.g. 'pkl_' for pkl_full.json etc.")
    args = ap.parse_args()
    d = args.results_dir
    p = args.prefix

    audit = load(f"{d}/audit.json")
    full  = load(f"{d}/{p}full.json")
    tcr   = load(f"{d}/{p}tcr_only.json")
    epi   = load(f"{d}/{p}epi_only.json")

    missing = [n for n, r in
               [("full", full), ("tcr_only", tcr), ("epi_only", epi)] if r is None]
    if missing:
        print(f"Missing result files for: {', '.join(missing)}")
        prefix_hint = f"--prefix {p}" if p else ""
        print(f"Run train.py (or train_pkl.py) for those modes first. {prefix_hint}")
        return

    label = f"BLOSUM62 pkl ({p.rstrip('_')})" if p else "CNN on raw sequences"
    print("\n" + "=" * 72)
    print(f"ABLATION RESULTS -- epitope-disjoint test set  [{label}]")
    print("=" * 72)
    print(f"{'Condition':<34}{'Test AUROC':>14}{'Macro AUROC':>14}")
    print("-" * 72)
    rows = [
        ("Full (epitope + CDR3)", full),
        ("CDR3 only (epitope zeroed)", tcr),
        ("Epitope only (CDR3 zeroed)", epi),
    ]
    for name, r in rows:
        print(f"{name:<34}{r['test_auroc']:>14.4f}"
              f"{r['macro_auroc_per_epitope']:>14.4f}")
    print("-" * 72)

    contrib       = full["test_auroc"] - tcr["test_auroc"]
    macro_contrib = (full["macro_auroc_per_epitope"]
                     - tcr["macro_auroc_per_epitope"])

    print(f"\nEpitope contribution (full - CDR3 only):  {contrib:+.4f} AUROC")
    print(f"Same, on macro AUROC:                     {macro_contrib:+.4f}")

    if audit and not p:
        floor = audit["provenance_leakage"]["cdr3_only_ngram_auroc"]
        print(f"\nProvenance floor (n-gram on CDR3 string):  {floor:.4f}")
        print(f"Full model over that floor:                "
              f"{full['test_auroc'] - floor:+.4f}")
        ov = audit["cdr3_overlap"]
        print(f"CDR3 sequences shared between classes:     "
              f"{ov['shared_cdr3']:,} of {ov['unique_positive_cdr3']:,} positive")

    print("\n" + "=" * 72)
    # Signed, not absolute: a NEGATIVE contribution means adding the epitope
    # made things worse, which is noise, not evidence of a peptide signal.
    if contrib < 0.02:
        if contrib < -0.005:
            print(f"VERDICT: the epitope input is not load-bearing "
                  f"(and slightly harmful, {contrib:+.4f}).")
        else:
            print("VERDICT: the epitope input is not load-bearing.")
        print()
        print("Zeroing the peptide costs the model essentially nothing.")
        print("Epitope-side counterfactuals against this model would map")
        print("score noise, not a peptide decision boundary.")
        print()
        print("This is a result, not a failure -- it is the diagnostic working.")
    else:
        print(f"VERDICT: the epitope input contributes {contrib:+.4f} AUROC.")
        print("Large enough to be worth probing with counterfactuals.")
    print("=" * 72 + "\n")

    out = f"{d}/{p}summary.json"
    summary = {
        "full": full,
        "tcr_only": tcr,
        "epi_only": epi,
        "epitope_contribution_auroc": round(contrib, 4),
        "epitope_contribution_macro": round(macro_contrib, 4),
        "provenance_floor": (
            audit["provenance_leakage"]["cdr3_only_ngram_auroc"]
            if audit and not p else None
        ),
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()