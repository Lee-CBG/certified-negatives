"""Reproduce catELMo's BLOSUM62 baseline encoding.

The encoding CODE is not in the repo, but the METHOD is fully specified in the
paper's Methods section (bioRxiv v2 / eLife reviewed preprint, "Comparison
studies > BLOSUM62"):

  * map each amino acid to a vector of length 24 via the BLOSUM62 matrix
    (the full 24-letter NCBI matrix: 20 AAs + B, Z, X, *)
  * predefined lengths: 20 for TCRs, 22 for epitopes (taken from ATM-TCR)
  * pad with the IMGT method -- if the sequence is SHORTER than the target,
    insert zero-padding IN THE MIDDLE; if LONGER, delete residues FROM THE
    MIDDLE until it fits
  * flatten: TCR -> 20 x 24 = 480 dims, epitope -> 22 x 24 = 528 dims

The middle padding matters. Naive right-padding puts every residue at a
different index for sequences of different lengths; IMGT-style middle padding
aligns the conserved N- and C-terminal ends of the CDR3 loop, which is the
whole reason the convention exists.

Output format is pinned by how bap.py consumes it:

    X1 = np.array(dat.tcr_embeds.to_list())   # TCR     -> inputA
    X2 = np.array(dat.epi_embeds.to_list())   # epitope -> inputB
    inputA = Input(shape=(len(X1_train[0]),))

so each *_embeds cell must be a FLAT 1-D vector, and the DataFrame needs
columns epi, tcr, binding, tcr_embeds, epi_embeds.

Usage:
    python3 src/blosum62_catelmo.py --csv data/TCREpitopePairs.csv \
        --out data/BLOSUM62.pkl
"""

import argparse

import numpy as np
import pandas as pd

# NCBI BLOSUM62 alphabet order, 24 letters.
ALPHABET = "ARNDCQEGHILKMFPSTWYVBZX*"
AA_INDEX = {a: i for i, a in enumerate(ALPHABET)}

_BLOSUM62 = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4],
    [-2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4],
    [-1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4],
    [ 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4],
    [-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1],
]

BLOSUM62 = np.array(_BLOSUM62, dtype=np.float32)
assert BLOSUM62.shape == (24, 24), "BLOSUM62 must be 24x24"
assert np.array_equal(BLOSUM62, BLOSUM62.T), "BLOSUM62 must be symmetric"


def imgt_fit(seq: str, target: int, left_bias: bool = True) -> str:
    """Fit a sequence to `target` length by padding or deleting in the MIDDLE.

    Short sequences get '-' inserted at the midpoint (encoded as zeros later).
    Long sequences have middle residues deleted.

    The paper does not state the tie-break for odd splits. `left_bias=True`
    gives the extra residue to the N-terminal half, which is the more common
    convention. Flip it with --right-bias if you need to match a specific
    reference implementation.
    """
    n = len(seq)
    if n == target:
        return seq
    if n < target:
        left = (n + 1) // 2 if left_bias else n // 2
        return seq[:left] + "-" * (target - n) + seq[left:]
    left = (target + 1) // 2 if left_bias else target // 2
    right = target - left
    return seq[:left] + (seq[-right:] if right else "")


def encode_flat(seq: str, target: int, left_bias: bool = True) -> np.ndarray:
    """IMGT-fit, then BLOSUM62 rows, then flatten to target*24."""
    fitted = imgt_fit(seq.upper(), target, left_bias)
    mat = np.zeros((target, 24), dtype=np.float32)
    for i, ch in enumerate(fitted):
        idx = AA_INDEX.get(ch)
        if idx is not None:          # '-' and unknowns stay all-zero
            mat[i] = BLOSUM62[idx]
    return mat.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/BLOSUM62.pkl")
    ap.add_argument("--tcr-len", type=int, default=20)
    ap.add_argument("--epi-len", type=int, default=22)
    ap.add_argument("--right-bias", action="store_true",
                    help="give the odd residue to the C-terminal half instead")
    args = ap.parse_args()
    left_bias = not args.right_bias

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["epi", "tcr", "binding"])
    df["epi"] = df["epi"].astype(str).str.strip().str.upper()
    df["tcr"] = df["tcr"].astype(str).str.strip().str.upper()
    df["binding"] = df["binding"].astype(int)

    for col, cap in [("tcr", args.tcr_len), ("epi", args.epi_len)]:
        L = df[col].str.len()
        print(f"{col:4s} len: min {L.min()}, median {int(L.median())}, "
              f"max {L.max()} | middle-deleted at {cap}: "
              f"{100 * (L > cap).mean():.3f}%")

    print("\nEncoding (IMGT middle padding, BLOSUM62 24-dim)...")
    df["tcr_embeds"] = [encode_flat(s, args.tcr_len, left_bias) for s in df["tcr"]]
    df["epi_embeds"] = [encode_flat(s, args.epi_len, left_bias) for s in df["epi"]]

    out = df[["epi", "tcr", "binding", "tcr_embeds", "epi_embeds"]]
    out.to_pickle(args.out)

    d1, d2 = len(out["tcr_embeds"].iloc[0]), len(out["epi_embeds"].iloc[0])
    print(f"\nWrote {args.out}  ({len(out):,} rows)")
    print(f"  tcr_embeds {d1} (= {args.tcr_len} x 24)   expected 480")
    print(f"  epi_embeds {d2} (= {args.epi_len} x 24)   expected 528")


if __name__ == "__main__":
    main()