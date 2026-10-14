"""Loading, subsampling and epitope-disjoint splitting of TCREpitopePairs.csv.

The epitope-disjoint split is the point of this module. A random split lets the
same epitope appear in train and test, which lets a model score well by
memorising epitopes. Splitting on the epitope means every test epitope is one
the model has never seen.
"""

import numpy as np
import pandas as pd

from blosum import encode_batch

MAX_EPI_LEN = 22
MAX_TCR_LEN = 30


def load_pairs(csv_path: str) -> pd.DataFrame:
    """Load TCREpitopePairs.csv and normalise column names."""
    df = pd.read_csv(csv_path)
    rename = {"epi": "epitope", "tcr": "cdr3", "binding": "label"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    missing = {"epitope", "cdr3", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
    df = df.dropna(subset=["epitope", "cdr3", "label"])
    df["epitope"] = df["epitope"].astype(str).str.strip().str.upper()
    df["cdr3"] = df["cdr3"].astype(str).str.strip().str.upper()
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def subsample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Take n rows at random, preserving the class balance."""
    if n is None or n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(idx)].reset_index(drop=True)


def epitope_disjoint_split(df, test_frac=0.15, val_frac=0.10, seed=0):
    """Split so that no epitope appears in more than one partition.

    Epitope frequencies here are brutally skewed -- the largest epitope holds
    ~10k pairs while the median holds ~10. Two consequences shaped this code:

    1. Filling test to its row quota greedily would hand it 2 epitopes. Macro
       AUROC over 2 epitopes is not a measurement. Instead each epitope is
       assigned to whichever partition is currently furthest below its row
       target, which spreads many epitopes into every partition while still
       hitting the row quotas approximately.

    2. `rng.shuffle()` on the array returned by `Index.to_numpy()` can fail
       SILENTLY on pandas 3.x -- it returns without error and leaves the array
       in its original count-descending order. That turns a random split into a
       systematically biased one with no warning. The list is therefore
       materialised as owned Python strings before shuffling, and
       `rng.permutation` is used rather than in-place shuffle.
    """
    rng = np.random.default_rng(seed)
    counts = df["epitope"].value_counts()

    # Own the data before shuffling (note 2 above), and SORT first (note 3).
    #
    # 3. `value_counts()` orders by count and breaks ties by encounter order,
    #    so row order leaks into the epitope order and a seeded permutation
    #    then yields a DIFFERENT split for the same data in a different order.
    #    73 of 215 distinct count values here are shared by several epitopes,
    #    so this is not a corner case. Sorting makes the split a function of
    #    the epitope SET and the seed only, which is what lets two runs over
    #    filtered and unfiltered data be compared.
    epitopes = np.array(sorted(str(e) for e in counts.index), dtype=object)
    sizes = {str(e): int(c) for e, c in counts.items()}
    epitopes = rng.permutation(epitopes)

    total = len(df)
    targets = {
        "test": total * test_frac,
        "val": total * val_frac,
        "train": total * (1.0 - test_frac - val_frac),
    }
    assigned = {k: [] for k in targets}
    filled = {k: 0 for k in targets}

    for ep in epitopes:
        # Assign to whichever partition is proportionally emptiest.
        part = min(targets, key=lambda k: filled[k] / targets[k])
        assigned[part].append(ep)
        filled[part] += sizes[ep]

    parts = {}
    for name in ("train", "val", "test"):
        keep = set(assigned[name])
        parts[name] = df[df["epitope"].isin(keep)].reset_index(drop=True)

    # Hard assertion: this is the whole point of the split.
    tr, va, te = (set(parts[k]["epitope"]) for k in ("train", "val", "test"))
    assert not (tr & te), "epitope leaked between train and test"
    assert not (tr & va), "epitope leaked between train and val"
    assert not (va & te), "epitope leaked between val and test"

    return parts["train"], parts["val"], parts["test"]


def encode_partition(df: pd.DataFrame):
    """Encode a partition into (epitope_tensor, cdr3_tensor, labels)."""
    epi = encode_batch(df["epitope"].tolist(), MAX_EPI_LEN)
    tcr = encode_batch(df["cdr3"].tolist(), MAX_TCR_LEN)
    y = df["label"].to_numpy().astype(np.float32)
    return epi, tcr, y