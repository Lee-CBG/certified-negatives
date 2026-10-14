#!/usr/bin/env python3
"""
run_external.py -- CPU-only diagnostic battery on an external TCR-epitope dataset.

Runs:
  1. Dataset statistics and overlap audit vs catELMo
  2. Provenance floor (n-gram TF-IDF LR on CDR3 only, epitope never shown)
  3. Home-size channel probe (requires negatives)

Usage:
  python3 src/run_external.py \
      --csv   data/immrep23_canon.csv  \
      --name  immrep23                 \
      --ref   data/TCREpitopePairs.csv \
      --out   results/ext_immrep23.json

  python3 src/run_external.py \
      --csv   data/vdjdb_matched.csv   \
      --name  vdjdb_matched            \
      --ref   data/TCREpitopePairs.csv \
      --out   results/ext_vdjdb_matched.json
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

try:
    from external import (
        load_catelmo,
        overlap_audit,
        provenance_floor,
        home_size_channel,
        positive_only_stats,
    )
except ImportError as e:
    print(f"ERROR: {e}\nIs PYTHONPATH=src set?", file=sys.stderr)
    sys.exit(1)


BAR = "─" * 64


def section(title):
    print(f"\n{BAR}\n  {title}\n{BAR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       required=True,
                    help="Canonical CSV with columns epi,tcr[,binding]")
    ap.add_argument("--name",      required=True,
                    help="Short label used in output")
    ap.add_argument("--ref",       required=True,
                    help="catELMo TCREpitopePairs.csv for overlap audit")
    ap.add_argument("--out",       required=True,
                    help="Output JSON path")
    ap.add_argument("--seeds",     nargs="+", type=int, default=[0, 1, 2],
                    help="Seeds for provenance floor (default: 0 1 2)")
    ap.add_argument("--ngram-max", type=int, default=4,
                    help="Max n-gram length for TF-IDF (default 4)")
    args = ap.parse_args()

    results = {"name": args.name, "csv": args.csv}
    t0 = time.time()

    # ── 1. load ───────────────────────────────────────────────────────────────
    section("Loading")
    df = pd.read_csv(args.csv)
    print(f"  Rows: {len(df):,}  columns: {list(df.columns)}")
    if "binding" not in df.columns:
        print("  NOTE: no 'binding' column — treating all rows as positives")
        df["binding"] = 1

    n_pos = int((df["binding"] == 1).sum())
    n_neg = int((df["binding"] == 0).sum())
    results.update({
        "n_rows": len(df), "n_pos": n_pos, "n_neg": n_neg,
        "n_epitopes": int(df["epi"].nunique()),
        "n_tcrs":     int(df["tcr"].nunique()),
    })
    print(f"  Positives: {n_pos:,}  Negatives: {n_neg:,}")
    print(f"  Epitopes:  {results['n_epitopes']:,}  TCRs: {results['n_tcrs']:,}")

    # ── 2. overlap audit ──────────────────────────────────────────────────────
    section("Overlap audit vs catELMo")
    ref   = load_catelmo(args.ref)
    audit = overlap_audit(df, ref)
    results["overlap_audit"] = audit
    print(f"  TCR  overlap: {audit['tcr_overlap']:>6,} / {results['n_tcrs']:>6,}"
          f"  ({audit['tcr_overlap_frac']:.1%})")
    print(f"  Epi  overlap: {audit['epi_overlap']:>6,} / {results['n_epitopes']:>6,}"
          f"  ({audit['epi_overlap_frac']:.1%})")
    print(f"  Pair overlap: {audit['pair_overlap']:>6,}")
    print(f"  Pos pair overlap: {audit['pos_pair_overlap']:>6,}")
    print()
    if audit["tcr_overlap_frac"] > 0.50:
        print("  ⚠  >50% of TCRs appear in catELMo — heavy contamination.")
        print("     Provenance floor will reflect shared pool statistics,")
        print("     not generalisation.")
    elif audit["tcr_overlap_frac"] > 0.10:
        print("  ⚠  >10% TCR overlap — treat floor as a lower bound on contamination.")
    else:
        print("  ✓  TCR overlap low — floor is a clean generalisation probe.")

    # ── 3. provenance floor ───────────────────────────────────────────────────
    if n_neg > 0:
        section("Provenance floor  (n-gram TF-IDF LR, CDR3 only)")
        floor_runs = []
        for seed in args.seeds:
            t1 = time.time()
            fr = provenance_floor(df, seed=seed,
                                  ngram_range=(1, args.ngram_max))
            fr["seed"] = seed
            floor_runs.append(fr)
            print(f"  seed {seed}: pooled {fr['pooled_auroc']:.4f} | "
                  f"macro {fr['macro_auroc']:.4f} ± {fr['macro_sd']:.4f} "
                  f"(n={fr['macro_n']})  [{time.time()-t1:.0f}s]")

        macro_vals = [r["macro_auroc"] for r in floor_runs
                      if not np.isnan(r["macro_auroc"])]
        floor_mean = float(np.mean(macro_vals)) if macro_vals else float("nan")
        floor_sd   = float(np.std(macro_vals))  if macro_vals else float("nan")
        results["provenance_floor"] = floor_runs
        results["floor_macro_mean"] = floor_mean
        results["floor_macro_sd"]   = floor_sd
        print(f"\n  Floor macro: {floor_mean:.4f} ± {floor_sd:.4f}")
        print()
        if floor_mean > 0.68:
            print("  ⚠  Floor >0.68 — same level as catELMo provenance signal.")
            print("     Negative construction likely carries similar pool-of-origin leak.")
        elif floor_mean > 0.60:
            print("  ⚠  Floor >0.60 — partial provenance signal present.")
        elif abs(floor_mean - 0.50) < 0.02:
            print("  ✓  Floor ≈ 0.50 — no detectable provenance signal.")
        else:
            print(f"  Floor = {floor_mean:.4f} — interpret relative to catELMo baseline (0.73).")
    else:
        section("Provenance floor — SKIPPED (no negatives)")
        results["provenance_floor"] = None
        results["floor_macro_mean"] = None

    # ── 4. home-size channel ──────────────────────────────────────────────────
    if n_neg > 0:
        section("Home-size channel probe")
        hs = home_size_channel(df)
        results["home_size"] = hs
        if hs:
            print(f"  home_max pooled: {hs['home_max_pooled']:.4f}")
            print(f"  home_max macro:  {hs['home_max_macro']:.4f}"
                  f" ± {hs['home_max_macro_sd']:.4f}"
                  f"  (n={hs['n_epitopes_evaluated']})")
            print(f"  within-epi gap (mean): {hs['within_epi_gap_mean']}")
            print()
            m = hs["home_max_macro"]
            if m < 0.20:
                print("  ⚠  home_max macro <0.20 — strong anti-predictive channel.")
                print("     Same Simpson reversal as in catELMo swap data.")
                print("     Matched construction is needed before any model training.")
            elif m < 0.45:
                print("  ⚠  home_max macro <0.45 — moderate anti-predictive channel.")
            elif abs(m - 0.50) < 0.03:
                print("  ✓  home_max macro ≈ 0.50 — channel absent (or construction already matched).")
            else:
                print(f"  home_max macro = {m:.4f}")
    else:
        section("Home-size channel — SKIPPED (no negatives)")
        results["home_size"] = None

    # ── 5. positive-only stats ────────────────────────────────────────────────
    if n_neg == 0:
        section("Positive-only dataset statistics")
        ps = positive_only_stats(df)
        results["positive_only_stats"] = ps
        print(f"  Pairs: {ps['n_pairs']:,}  Epitopes: {ps['n_epitopes']:,}  "
              f"TCRs: {ps['n_tcrs']:,}")
        print(f"  Median pairs/epitope: {ps['median_pairs_per_epi']:.1f}  "
              f"Max: {ps['max_pairs_per_epi']:,}")
        print(f"  Singletons: {ps['epitopes_with_1_pair']:,} epitopes with 1 pair")
        print(f"  Top 10 epitopes: {ps['top10_epitopes']}")

    # ── done ──────────────────────────────────────────────────────────────────
    results["elapsed_s"] = round(time.time() - t0, 1)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n{BAR}")
    print(f"  Written: {args.out}  ({results['elapsed_s']}s)")
    print(BAR)


if __name__ == "__main__":
    main()