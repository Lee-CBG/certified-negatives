"""Rung 6: rebuild the negatives by TCR swap.

Unlike the other rungs this is not a filter. It discards the original negatives
entirely and generates new ones from the positives, following the protocol used
by IMMREP22, IMMREP23 and Liao et al.

    For each epitope e with n positive pairs, emit n negative pairs by taking
    receptors observed with OTHER epitopes and pairing them with e.

Two constraints, both of which matter:

  1. Donor epitopes must be at Levenshtein distance > 3 from the target.
     Closely similar peptides are the ones most likely to be genuinely
     cross-reactive, so a receptor binding a near neighbour of e is a poor
     bet as a non-binder of e. This is the IMMREP23 threshold.

  2. Receptor usage is balanced, not uniform-random. If receptors are drawn
     uniformly, some are used as negative partners several times and others
     never, and a receptor never used as a negative appears only in the
     positive class. IMMREP23 reported exactly this failure: the twenty most
     frequent TCRs in their test set were all negatives, while every TCR
     appearing once was a positive. The assignment here draws from a pool
     where each receptor appears as often as it does among the positives,
     which balances usage in expectation and is checked afterwards.

The expectation is that the provenance floor collapses toward 0.5, because
every receptor now appears in both classes and the receptor string alone
carries no label information. If it does NOT collapse, that is the more
interesting result and should be chased rather than explained away.
"""

import numpy as np
import pandas as pd

from data import epitope_disjoint_split


def levenshtein(a: str, b: str) -> int:
    """Plain DP edit distance. Sequences here are short, so this is fine."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _too_similar(epitopes, threshold=3, verbose=True):
    """Map each epitope to the set of epitopes within `threshold` edits."""
    eps = list(epitopes)
    near = {e: set() for e in eps}
    for i, a in enumerate(eps):
        for b in eps[i + 1:]:
            # cheap length prefilter before the DP
            if abs(len(a) - len(b)) > threshold:
                continue
            if levenshtein(a, b) <= threshold:
                near[a].add(b)
                near[b].add(a)
    if verbose:
        n = sum(len(v) for v in near.values()) // 2
        print(f"  {n:,} epitope pairs within {threshold} edits "
              f"(excluded as donors)")
    return near


def tcr_swap_negatives(df, seed=0, min_distance=3, verbose=True):
    """Replace the negatives with swapped ones built from the positives."""
    rng = np.random.default_rng(seed)
    pos = df[df.label == 1][["epitope", "cdr3"]].copy()
    if verbose:
        print(f"  building from {len(pos):,} positive pairs, "
              f"{pos.epitope.nunique()} epitopes")

    epitopes = sorted(pos["epitope"].unique())
    near = _too_similar(epitopes, min_distance, verbose)

    # Which epitopes each receptor was observed with -- never reuse those.
    observed = pos.groupby("cdr3")["epitope"].agg(set).to_dict()

    # Donor pool: every receptor occurrence among the positives. Drawing from
    # this pool means a receptor's expected count as a negative partner equals
    # its count as a positive, which is what balances the classes.
    pool = pos["cdr3"].to_numpy().copy()
    pool = rng.permutation(pool)

    need = pos["epitope"].value_counts().to_dict()
    neg_rows = []
    cursor = 0
    misses = 0

    for ep in epitopes:
        n = need[ep]
        blocked = near[ep] | {ep}
        taken = 0
        attempts = 0
        while taken < n and attempts < n * 40:
            if cursor >= len(pool):
                pool = rng.permutation(pool)
                cursor = 0
            cdr3 = pool[cursor]
            cursor += 1
            attempts += 1
            obs = observed.get(cdr3, set())
            # reject if this receptor binds the target or any near neighbour
            if obs & blocked:
                continue
            neg_rows.append((ep, cdr3))
            taken += 1
        misses += n - taken

    neg = pd.DataFrame(neg_rows, columns=["epitope", "cdr3"])
    neg["label"] = 0
    out = pd.concat([pos.assign(label=1), neg], ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)

    if verbose:
        if misses:
            print(f"  WARNING: {misses:,} negatives could not be placed "
                  f"under the constraints")
        print(f"  result: {len(out):,} pairs "
              f"({100 * out.label.mean():.2f}% positive)")
    return out


def tcr_swap_within_partitions(df, seed=0, min_distance=3, verbose=True):
    """Build swap negatives SEPARATELY inside each partition.

    Why this exists. Building negatives globally and splitting afterwards
    inverts the label for any receptor whose positive epitope lands in test:
    its positive pairing is held out, but it is still drawn from the global
    donor pool and so appears in train only as a negative. The model learns
    "this receptor means negative" and test then asks about it as a positive.
    Measured on this dataset, that produces a correlation of -0.96 between a
    receptor's train label rate and its test label rate, across 81% of test
    receptors, and a receptor-only floor of 0.27 -- further from chance than
    the untouched dataset.

    Splitting first and generating each partition's negatives from that
    partition's own receptors means no receptor crosses the boundary, so the
    inverted signal cannot form.

    Returns (train, val, test), already built. Note these are NOT a split of a
    single frame: each partition has its own generated negatives.
    """
    parts = epitope_disjoint_split(df, seed=seed)
    names = ("train", "val", "test")
    out = []
    for name, part in zip(names, parts):
        if verbose:
            print(f"  [{name}] ", end="")
        built = tcr_swap_negatives(part, seed=seed,
                                   min_distance=min_distance, verbose=False)
        if verbose:
            print(f"{len(part):,} in -> {len(built):,} pairs, "
                  f"{built.epitope.nunique()} epitopes, "
                  f"{100 * built.label.mean():.1f}% positive")
        out.append(built)

    if verbose:
        rec = [set(p["cdr3"]) for p in out]
        print(f"  receptor overlap train/test: "
              f"{len(rec[0] & rec[2]):,} sequences")
    return tuple(out)


def verify_swap(df, verbose=True):
    """Check the two balance properties the construction is supposed to give."""
    ep = df.groupby("epitope")["label"].mean()
    rc = df.groupby("cdr3")["label"].agg(["mean", "size"])
    multi = rc[rc["size"] > 1]

    report = {
        "epitope_positive_rate_min": round(float(ep.min()), 4),
        "epitope_positive_rate_max": round(float(ep.max()), 4),
        "epitopes_off_50pct_by_1pct": int((ep.sub(0.5).abs() > 0.01).sum()),
        "n_epitopes": int(len(ep)),
        "receptors_seen_once": int((rc["size"] == 1).sum()),
        "receptors_seen_multiple": int(len(multi)),
        "multi_receptors_all_positive": int((multi["mean"] == 1.0).sum()),
        "multi_receptors_all_negative": int((multi["mean"] == 0.0).sum()),
    }
    if verbose:
        print("  VERIFY")
        print(f"    epitope positive rate: "
              f"{report['epitope_positive_rate_min']:.4f} to "
              f"{report['epitope_positive_rate_max']:.4f}")
        print(f"    epitopes more than 1pp off 50%: "
              f"{report['epitopes_off_50pct_by_1pct']} / {report['n_epitopes']}")
        print(f"    receptors appearing once: {report['receptors_seen_once']:,}")
        print(f"    receptors appearing 2+ times: "
              f"{report['receptors_seen_multiple']:,}, of which "
              f"{report['multi_receptors_all_positive']:,} all-positive and "
              f"{report['multi_receptors_all_negative']:,} all-negative")
        print("    (receptors appearing once are unavoidably single-class;")
        print("     what matters is that repeated ones are mixed)")
    return report