"""Build encoded pkls using the home-size-MATCHED swap negatives.

WHY

fix_swap_pool.py validated the matched draw on the raw sequences, 3 seeds:

    test partition, home_max per-epitope AUROC   0.0837 -> 0.4816
    median within-epitope gap                    -2632  -> ~0
    positive rate                                0.5000 (unchanged)
    unplaced slots                               0

That is a proxy. The question it stands in for is whether a receptor-only model
returns to ~0.50 macro, and answering that needs encoded data and a training run.
This script produces the data.

It imports tcr_swap_negatives_matched from fix_swap_pool rather than editing
tcr_swap.py, so the existing construction stays intact and the two can be
compared side by side. Port it into tcr_swap.py only after the model run
confirms it does what the proxy says.

KNOWN RESIDUAL, carried forward deliberately

Matching is impossible for the very largest epitopes: in a test partition holding
one 5,793-pair epitope and nothing else near that size, every same-bucket donor
is one of its own receptors and therefore blocked, so its whole quota falls back
to smaller buckets. Measured: 0 fallbacks in train (713 epitopes, everything has
company), 7,439-9,878 in test. Train comes out at pooled 0.5048, test at 0.5682.

So the matched construction is clean where the epitope-size distribution has
enough density and approximate at the top of it. Report that rather than hiding
it -- and check whether the residual tracks epitope size in the model run.

USAGE

    export PYTHONPATH=src
    python3 src/make_matched_pkl.py --csv data/TCREpitopePairs.csv \
        --seed 0 --out data/matched_seed0.pkl

Then make_grouped_pkl.py turns it into the seen-epitope control, exactly as it
did for the swap pkls, and train_pkl.py --split partition consumes either.

About three minutes per seed; the Levenshtein pass over epitope pairs dominates.
Each pkl is ~1.24 GB.
"""

import argparse
import os

import pandas as pd

from blosum62_catelmo import encode_flat
from data import epitope_disjoint_split, load_pairs, subsample
from sweep_granularity import build_bucketed
from tcr_swap import verify_swap

TCR_LEN = 20
EPI_LEN = 22


def encode_frame(df):
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=300016)
    ap.add_argument("--min-distance", type=int, default=3)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="bucket = floor(log2(home_max) * scale). "
                         "4.0 is the confirmed setting (seeds 3-5).")
    args = ap.parse_args()

    df = subsample(load_pairs(args.csv), args.max_rows, seed=0)
    print(f"loaded {len(df):,} pairs\n")

    # Partitions BEFORE negatives, each partition's negatives from its own
    # receptors -- the §4.5 fix, unchanged. Only the draw within that is new.
    parts = epitope_disjoint_split(df, seed=args.seed)
    frames = []
    for name, part in zip(("train", "val", "test"), parts):
        print(f"[{name}] {len(part):,} rows, {part.epitope.nunique()} epitopes")
        built, bstats = build_bucketed(
            part, seed=args.seed, min_distance=args.min_distance,
            scale=args.scale)
        built = built[['epitope', 'cdr3', 'label']]
        print(f'    scale={args.scale:g}  {bstats}')
        verify_swap(built)
        enc = encode_frame(built)
        enc["partition"] = name
        frames.append(enc)
        print()

    out = pd.concat(frames, ignore_index=True)

    # Interleave positives and negatives. This is the guard against the float32
    # tie-breaking artifact that once read epi_only macro as 0.5624 on a
    # label-ordered file (§7).
    out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_pickle(args.out)
    print(f"wrote {args.out}")
    print(f"  rows           {len(out):,}")
    print(f"  positive rate  {out.binding.mean():.4f}")
    print(f"  tcr dim        {len(out.tcr_embeds.iloc[0])}")
    print(f"  epi dim        {len(out.epi_embeds.iloc[0])}")
    for name, g in out.groupby("partition"):
        print(f"  {name:<6} {len(g):>8,} pairs, "
              f"{g.epi.nunique():>4} epitopes, "
              f"{100 * g.binding.mean():.1f}% positive")

    rec = {n: set(g.tcr) for n, g in out.groupby("partition")}
    print(f"\n  receptor overlap train/test: "
          f"{len(rec['train'] & rec['test']):,} sequences "
          f"(low thousands expected; a large number means partition-local "
          f"construction failed)")


if __name__ == "__main__":
    main()