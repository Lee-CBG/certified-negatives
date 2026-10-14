"""Interventions that remove one shortcut at a time from the dataset.

Each function takes a DataFrame with columns epitope / cdr3 / label and returns
a modified one. They are meant to be applied cumulatively as rungs of a ladder,
measuring the provenance floor after each, to answer: how much of the floor was
that particular artifact, and what survives when all of them are gone?

The interventions in order of how uncontroversial they are:

  repair_malformed   A real TCR-beta CDR3 starts with C and ends with F.
                     4.67% of rows here violate that, and 94.3% of those are
                     positives, so the negative file is clean and the positive
                     file has a preprocessing defect. Fixing this is not a
                     modelling choice, it is a bug fix.

  trim_framework     The conserved V and J framework residues carry the least
                     specificity information of any part of the receptor, yet
                     CASS / CAS / CA are the largest features in the floor
                     model. Removing them forces any model onto the
                     hypervariable core.

  match_length       Positives are 0.727 residues shorter on average, and
                     length alone reaches AUROC 0.57-0.64 depending on split.
                     Resampling to equalise the length distributions removes
                     that channel.

  match_vj_proxy     V and J usage differ between the pools (CASS at 63.2%
                     vs 58.4%, GYTF endings at 6.5% vs 10.8%). Without V/J
                     gene annotations the terminal k-mers are the available
                     proxy, so this matches on those.

Unlike the first three, match_vj_proxy is arguably removing biology rather
than an artifact, which is exactly why it belongs at the top of the ladder and
should be reported separately.
"""

import numpy as np
import pandas as pd


def _report(name, before, after):
    lost = before - after
    print(f"  {name:<22} {before:>8,} -> {after:>8,}  ({lost:,} dropped, "
          f"{100 * lost / max(before, 1):.2f}%)")


def is_malformed(cdr3: pd.Series) -> pd.Series:
    """True where the sequence lacks the conserved C start or F end."""
    return ~cdr3.str.startswith("C") | ~cdr3.str.endswith("F")


def repair_malformed(df, mode="drop", verbose=True):
    """Rung 1. Handle sequences that are not well-formed CDR3s.

    mode="drop"   remove them entirely (safest, loses ~4.7% of rows)
    mode="repair" add the missing C and/or F back (keeps rows, assumes the
                  defect really is a trimming inconsistency)
    """
    before = len(df)
    bad = is_malformed(df["cdr3"])
    out = df.copy()

    if mode == "drop":
        out = out[~bad].reset_index(drop=True)
    elif mode == "repair":
        s = out["cdr3"]
        s = s.where(s.str.startswith("C"), "C" + s)
        s = s.where(s.str.endswith("F"), s + "F")
        out["cdr3"] = s
    else:
        raise ValueError(f"unknown mode: {mode}")

    if verbose:
        _report(f"repair_malformed[{mode}]", before, len(out))
    return out


def trim_framework(df, n_start=1, n_end=1, min_len=4, verbose=True):
    """Rung 2. Strip conserved framework residues from both ends.

    n_start=1, n_end=1 removes the leading C and trailing F.
    n_start=3, n_end=3 removes the CAS.../...YF framework more aggressively,
    leaving roughly the hypervariable core.

    Rows whose remaining core is shorter than min_len are dropped.
    """
    before = len(df)
    out = df.copy()
    end = -n_end if n_end > 0 else None
    out["cdr3"] = df["cdr3"].str[n_start:end]
    out = out[out["cdr3"].str.len() >= min_len].reset_index(drop=True)
    if verbose:
        _report(f"trim_framework[{n_start},{n_end}]", before, len(out))
    return out


def match_length(df, seed=0, verbose=True):
    """Rung 3. Resample so both classes share a length distribution.

    For each length, keep an equal number of positives and negatives by
    downsampling the larger side. Removes length as a usable channel.
    """
    before = len(df)
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["_len"] = out["cdr3"].str.len()

    keep = []
    for L, g in out.groupby("_len"):
        pos = g[g.label == 1]
        neg = g[g.label == 0]
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        keep.append(pos.iloc[rng.choice(len(pos), k, replace=False)])
        keep.append(neg.iloc[rng.choice(len(neg), k, replace=False)])

    out = pd.concat(keep).drop(columns="_len").reset_index(drop=True)
    if verbose:
        _report("match_length", before, len(out))
    return out


def match_vj_proxy(df, k=4, seed=0, verbose=True):
    """Rung 4. Match the classes on terminal k-mers as a V/J proxy.

    Without V and J gene annotations, the first and last k residues are the
    available stand-in. For each (prefix, suffix) combination, keep equal
    numbers from both classes.

    Caveat: this may be removing real biology rather than an artifact, since
    V/J usage genuinely differs between repertoires. Report it separately from
    the lower rungs.
    """
    before = len(df)
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["_key"] = out["cdr3"].str[:k] + "|" + out["cdr3"].str[-k:]

    keep = []
    for _, g in out.groupby("_key"):
        pos = g[g.label == 1]
        neg = g[g.label == 0]
        n = min(len(pos), len(neg))
        if n == 0:
            continue
        keep.append(pos.iloc[rng.choice(len(pos), n, replace=False)])
        keep.append(neg.iloc[rng.choice(len(neg), n, replace=False)])

    if not keep:
        raise RuntimeError("match_vj_proxy removed everything; try smaller k")
    out = pd.concat(keep).drop(columns="_key").reset_index(drop=True)
    if verbose:
        _report(f"match_vj_proxy[k={k}]", before, len(out))
    return out


# Named rungs, applied cumulatively by run_ladder.py
LADDER = [
    ("0_baseline",      lambda d, seed: d),
    ("1_no_malformed",  lambda d, seed: repair_malformed(d, "drop")),
    ("2_trim_CF",       lambda d, seed: trim_framework(d, 1, 1)),
    ("3_trim_core",     lambda d, seed: trim_framework(d, 2, 2)),
    ("4_match_length",  lambda d, seed: match_length(d, seed)),
    ("5_match_vj",      lambda d, seed: match_vj_proxy(d, 3, seed)),
]