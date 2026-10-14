"""BLOSUM50 encoding for amino acid sequences.

The matrix is hardcoded so this repo has no biopython dependency and the
encoding is byte-for-byte reproducible on any machine.

Each amino acid becomes a 20-dim vector (its row in BLOSUM50, restricted to
the 20 standard amino acids). Padding positions are all-zero, so they
contribute nothing to convolutions.
"""

import numpy as np

AA = "ARNDCQEGHILKMFPSTWYV"
AA_INDEX = {a: i for i, a in enumerate(AA)}

# BLOSUM50, 20x20, row/col order = AA above.
_BLOSUM50 = [
    [ 5,-2,-1,-2,-1,-1,-1, 0,-2,-1,-2,-1,-1,-3,-1, 1, 0,-3,-2, 0],
    [-2, 7,-1,-2,-4, 1, 0,-3, 0,-4,-3, 3,-2,-3,-3,-1,-1,-3,-1,-3],
    [-1,-1, 7, 2,-2, 0, 0, 0, 1,-3,-4, 0,-2,-4,-2, 1, 0,-4,-2,-3],
    [-2,-2, 2, 8,-4, 0, 2,-1,-1,-4,-4,-1,-4,-5,-1, 0,-1,-5,-3,-4],
    [-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1],
    [-1, 1, 0, 0,-3, 7, 2,-2, 1,-3,-2, 2, 0,-4,-1, 0,-1,-1,-1,-3],
    [-1, 0, 0, 2,-3, 2, 6,-3, 0,-4,-3, 1,-2,-3,-1,-1,-1,-3,-2,-3],
    [ 0,-3, 0,-1,-3,-2,-3, 8,-2,-4,-4,-2,-3,-4,-2, 0,-2,-3,-3,-4],
    [-2, 0, 1,-1,-3, 1, 0,-2,10,-4,-3, 0,-1,-1,-2,-1,-2,-3, 2,-4],
    [-1,-4,-3,-4,-2,-3,-4,-4,-4, 5, 2,-3, 2, 0,-3,-3,-1,-3,-1, 4],
    [-2,-3,-4,-4,-2,-2,-3,-4,-3, 2, 5,-3, 3, 1,-4,-3,-1,-2,-1, 1],
    [-1, 3, 0,-1,-3, 2, 1,-2, 0,-3,-3, 6,-2,-4,-1, 0,-1,-3,-2,-3],
    [-1,-2,-2,-4,-2, 0,-2,-3,-1, 2, 3,-2, 7, 0,-3,-2,-1,-1, 0, 1],
    [-3,-3,-4,-5,-2,-4,-3,-4,-1, 0, 1,-4, 0, 8,-4,-3,-2, 1, 4,-1],
    [-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3],
    [ 1,-1, 1, 0,-1, 0,-1, 0,-1,-3,-3, 0,-2,-3,-1, 5, 2,-4,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 2, 5,-3,-2, 0],
    [-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1, 1,-4,-4,-3,15, 2,-3],
    [-2,-1,-2,-3,-3,-1,-2,-3, 2,-1,-1,-2, 0, 4,-3,-2,-2, 2, 8,-1],
    [ 0,-3,-3,-4,-1,-3,-3,-4,-4, 4, 1,-3, 1,-1,-3,-2, 0,-3,-1, 5],
]

BLOSUM50 = np.array(_BLOSUM50, dtype=np.float32)

# Scale to roughly unit variance. Keeps the CNN's first layer well-conditioned.
BLOSUM50_SCALED = BLOSUM50 / 5.0


def encode_sequence(seq: str, max_len: int) -> np.ndarray:
    """Encode one sequence as (max_len, 20) float32.

    Unknown characters and padding are all-zero rows. Sequences longer than
    max_len are truncated from the right.
    """
    out = np.zeros((max_len, 20), dtype=np.float32)
    for i, ch in enumerate(seq[:max_len]):
        idx = AA_INDEX.get(ch.upper())
        if idx is not None:
            out[i] = BLOSUM50_SCALED[idx]
    return out


def encode_batch(seqs, max_len: int) -> np.ndarray:
    """Encode a list of sequences as (n, max_len, 20) float32."""
    out = np.zeros((len(seqs), max_len, 20), dtype=np.float32)
    for n, seq in enumerate(seqs):
        out[n] = encode_sequence(seq, max_len)
    return out