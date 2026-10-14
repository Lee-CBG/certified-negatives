"""Run this FIRST to tell us what format the pkl is in.

    python3 src/inspect_pkl.py --pkl path/to/BLOSUM62.pkl

Prints column names, embedding dimensions, row count, and a few sanity checks.
The output of this script determines which train command to use next.
"""

import argparse
import sys
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="path to the BLOSUM62 pkl file")
    args = ap.parse_args()

    print(f"Loading {args.pkl} ...")
    df = pd.read_pickle(args.pkl)

    print(f"\n{'='*55}")
    print(f"  rows          {len(df):,}")
    print(f"  columns       {list(df.columns)}")
    print(f"{'='*55}\n")

    for col in df.columns:
        v = df[col].iloc[0]
        if isinstance(v, np.ndarray):
            print(f"  {col:15s}  numpy array  shape={v.shape}  dtype={v.dtype}"
                  f"  min={v.min():.2f}  max={v.max():.2f}")
        elif hasattr(v, "__len__"):
            print(f"  {col:15s}  {type(v).__name__}  len={len(v)}")
        else:
            print(f"  {col:15s}  {type(v).__name__}  = {repr(v)[:50]}")

    # Work out embedding dims
    tcr_col = next((c for c in df.columns if "tcr" in c.lower() and c != "tcr"), None)
    epi_col = next((c for c in df.columns if "epi" in c.lower() and c != "epi"), None)

    print()
    if tcr_col and epi_col:
        td = len(df[tcr_col].iloc[0])
        ed = len(df[epi_col].iloc[0])
        print(f"  TCR embed dim  {td}  ({td//20}×20 or {td//24}×24)")
        print(f"  epi embed dim  {ed}  ({ed//20}×20 or {ed//22}×22 or {ed//24}×24)")

        # Guess which variant this is
        if td == 480 and ed == 528:
            print("\n  >>> catELMo original: 24-col BLOSUM62, IMGT middle-padding")
            print("      TCR max_len=20 × 24 = 480")
            print("      epi max_len=22 × 24 = 528")
        elif td == 400 and ed == 440:
            print("\n  >>> Our script (blosum62_catelmo.py): 20-col BLOSUM62, right-padding")
            print("      TCR max_len=20 × 20 = 400")
            print("      epi max_len=22 × 20 = 440")
        else:
            print(f"\n  >>> Unknown variant — TCR dim {td}, epi dim {ed}")
            print("      Paste this output so we can write the right train command.")

    print(f"\n  positive rate  {df['binding'].mean():.4f}")
    print(f"  unique TCRs    {df['tcr'].nunique():,}")
    print(f"  unique epitopes {df['epi'].nunique():,}")

    print(f"\n{'='*55}")
    print("  Paste this output before running train_pkl.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()