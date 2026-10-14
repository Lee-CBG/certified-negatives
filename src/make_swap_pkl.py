"""Build a BLOSUM62-encoded pkl from either the original data or TCR-swap data.

Two modes, and you want BOTH so the comparison is like-for-like:

    --mode baseline   the original negatives, encoded with this script
    --mode swap       partition-local TCR-swap negatives, same encoder

The baseline pkl here will NOT reproduce the numbers from catELMo's own
BLOSUM62.pkl, because that one uses a 24-column matrix with IMGT middle-padding
and this uses 20 columns with right-padding. That is fine and it is why you
build both: the comparison that matters is swap vs baseline under an identical
encoder, not either one against a previously published figure.

Swap mode writes a `partition` column. The partitions are built BEFORE the
negatives, and each partition's negatives are drawn only from its own
receptors, so re-splitting afterwards would destroy the property the whole
construction exists to provide. train_pkl.py uses that column when present.

    python3 src/make_swap_pkl.py --csv data/TCREpitopePairs.csv \
        --mode swap --seed 0 --out data/swap_seed0.pkl
    python3 src/make_swap_pkl.py --csv data/TCREpitopePairs.csv \
        --mode baseline --seed 0 --out data/base_seed0.pkl
"""

import argparse

import numpy as np
import pandas as pd

from blosum62_catelmo import encode_flat
from data import epitope_disjoint_split, load_pairs, subsample
from tcr_swap import tcr_swap_within_partitions, verify_swap

TCR_LEN = 20
EPI_LEN = 22


def encode_frame(df, verbose=True):
    if verbose:
        print(f"  encoding {len(df):,} pairs ...")
    out = pd.DataFrame({
        "epi": df["epitope"].to_numpy(),
        "tcr": df["cdr3"].to_numpy(),
        "binding": df["label"].to_numpy().astype(int),
    })
    out["tcr_embeds"] = [encode_flat(s, TCR_LEN) for s in out["tcr"]]
    out["epi_embeds"] = [encode_flat(s, EPI_LEN) for s in out["epi"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mode", required=True, choices=["baseline", "swap"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=300016)
    ap.add_argument("--min-distance", type=int, default=3)
    args = ap.parse_args()

    df = subsample(load_pairs(args.csv), args.max_rows, seed=0)
    print(f"loaded {len(df):,} pairs\n")

    if args.mode == "swap":
        print("building partition-local TCR-swap negatives")
        tr, va, te = tcr_swap_within_partitions(
            df, seed=args.seed, min_distance=args.min_distance
        )
        print()
        for name, part in [("train", tr), ("val", va), ("test", te)]:
            print(f"  [{name}]")
            verify_swap(part)
        frames = []
        for name, part in [("train", tr), ("val", va), ("test", te)]:
            enc = encode_frame(part, verbose=False)
            enc["partition"] = name
            frames.append(enc)
        out = pd.concat(frames, ignore_index=True)
    else:
        print("using the original negatives")
        tr, va, te = epitope_disjoint_split(df, seed=args.seed)
        frames = []
        for name, part in [("train", tr), ("val", va), ("test", te)]:
            enc = encode_frame(part, verbose=False)
            enc["partition"] = name
            frames.append(enc)
        out = pd.concat(frames, ignore_index=True)

    # Shuffle so positives and negatives are interleaved. This is what avoids
    # the float32 tie-breaking artifact that made epi_only macro AUROC read
    # 0.5624 instead of 0.5 on a label-ordered file.
    out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out.to_pickle(args.out)
    print(f"\nwrote {args.out}")
    print(f"  rows           {len(out):,}")
    print(f"  positive rate  {out.binding.mean():.4f}")
    print(f"  tcr dim        {len(out.tcr_embeds.iloc[0])}")
    print(f"  epi dim        {len(out.epi_embeds.iloc[0])}")
    for name, g in out.groupby("partition"):
        print(f"  {name:<6} {len(g):>8,} pairs, "
              f"{g.epi.nunique():>4} epitopes, "
              f"{100 * g.binding.mean():.1f}% positive")


if __name__ == "__main__":
    main()