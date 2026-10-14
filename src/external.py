"""
external.py -- loaders and CPU-only diagnostics for external TCR-epitope datasets.

All public functions return canonical DataFrames with columns [epi, tcr, binding].
Import from anywhere with PYTHONPATH=src.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# ── column-name candidates ────────────────────────────────────────────────────

_IMMREP_CDR3 = ["CDR3b", "CDR3", "cdr3b", "cdr3", "TRB_CDR3", "beta",
                 "junction_aa", "CDR3_beta"]
_IMMREP_EPI  = ["Epitope", "epitope", "Antigen", "antigen", "peptide",
                 "Peptide", "antigen.epitope"]
_IMMREP_BIND = ["Label", "label", "Binding", "binding", "binder",
                 "Binder", "target"]

_VDJDB_CDR3  = ["cdr3", "CDR3", "cdr3.beta", "cdr3b", "junction_aa"]
_VDJDB_EPI   = ["antigen.epitope", "Epitope", "epitope", "Antigen"]
_VDJDB_GENE  = ["gene", "Gene"]
_VDJDB_SPEC  = ["species", "Species"]


def _pick(df, candidates, label=""):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Cannot find {label or 'column'} — tried: {candidates}\n"
        f"  Available columns: {sorted(df.columns.tolist())}"
    )


def _canonicalise(df, cdr3_col, epi_col, binding_col=None):
    out = pd.DataFrame({
        "epi":     df[epi_col].astype(str).str.strip().str.upper(),
        "tcr":     df[cdr3_col].astype(str).str.strip().str.upper(),
        "binding": df[binding_col].astype(int) if binding_col else 1,
    })
    out = out[out["epi"].str.len() >= 7]
    out = out[out["tcr"].str.len() >= 5]
    out = out[~out["epi"].isin(["NAN", "NA", "NONE", ""])]
    out = out[~out["tcr"].isin(["NAN", "NA", "NONE", ""])]
    return out.reset_index(drop=True)


# ── public loaders ────────────────────────────────────────────────────────────

def load_catelmo(path):
    df = pd.read_csv(path)
    df["epi"] = df["epi"].astype(str).str.strip().str.upper()
    df["tcr"] = df["tcr"].astype(str).str.strip().str.upper()
    return df[["epi", "tcr", "binding"]].dropna().reset_index(drop=True)


def load_immrep(path):
    """
    Load an IMMREP training TSV → canonical [epi, tcr, binding].
    Handles column-name variants across 2021-2023 challenge releases.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    print(f"  [load_immrep] {len(df):,} rows  columns: {list(df.columns)}")
    cdr3_col = _pick(df, _IMMREP_CDR3, "CDR3β")
    epi_col  = _pick(df, _IMMREP_EPI,  "epitope")
    bind_col = _pick(df, _IMMREP_BIND, "label")
    out = _canonicalise(df, cdr3_col, epi_col, bind_col)
    print(f"  [load_immrep] canonical {len(out):,} rows  "
          f"{int((out['binding']==1).sum())} pos / {int((out['binding']==0).sum())} neg")
    return out


def load_vdjdb(path):
    """
    Load VDJdb full or slim TSV → canonical [epi, tcr, binding=1].
    Filters to human TRB, deduplicates.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    print(f"  [load_vdjdb] {len(df):,} rows  columns: {list(df.columns)[:14]} ...")

    # TRB filter
    for g in _VDJDB_GENE:
        if g in df.columns:
            df = df[df[g].str.upper() == "TRB"]
            break

    # human filter
    for sc in _VDJDB_SPEC:
        if sc in df.columns:
            df = df[df[sc].astype(str).str.lower().str.contains(
                r"human|homo", na=False, regex=True)]
            break

    cdr3_col = _pick(df, _VDJDB_CDR3, "CDR3")
    epi_col  = _pick(df, _VDJDB_EPI,  "epitope")
    out = _canonicalise(df, cdr3_col, epi_col, binding_col=None)
    out = out.drop_duplicates(subset=["epi", "tcr"]).reset_index(drop=True)
    print(f"  [load_vdjdb] canonical {len(out):,} unique positive pairs")
    return out


# ── overlap audit ─────────────────────────────────────────────────────────────

def overlap_audit(df, ref):
    ref_tcrs  = set(ref["tcr"])
    ref_epis  = set(ref["epi"])
    ref_pairs = set(zip(ref["epi"], ref["tcr"]))
    df_tcrs   = set(df["tcr"])
    df_epis   = set(df["epi"])
    df_pairs  = set(zip(df["epi"], df["tcr"]))
    pos = df[df["binding"] == 1] if "binding" in df.columns else df
    pos_pairs = set(zip(pos["epi"], pos["tcr"]))
    return {
        "n_pairs":          int(len(df)),
        "n_pos":            int((df["binding"] == 1).sum()) if "binding" in df.columns else int(len(df)),
        "n_neg":            int((df["binding"] == 0).sum()) if "binding" in df.columns else 0,
        "n_epitopes":       int(df["epi"].nunique()),
        "n_tcrs":           int(df["tcr"].nunique()),
        "tcr_overlap":      int(len(df_tcrs & ref_tcrs)),
        "epi_overlap":      int(len(df_epis & ref_epis)),
        "pair_overlap":     int(len(df_pairs & ref_pairs)),
        "pos_pair_overlap": int(len(pos_pairs & ref_pairs)),
        "tcr_overlap_frac": round(len(df_tcrs & ref_tcrs) / max(len(df_tcrs), 1), 4),
        "epi_overlap_frac": round(len(df_epis & ref_epis) / max(len(df_epis), 1), 4),
    }


# ── provenance floor ──────────────────────────────────────────────────────────

def provenance_floor(df, seed=0, n_splits=5, ngram_range=(1, 4)):
    """
    Character n-gram TF-IDF logistic regression on CDR3 string alone.
    Primary metric: macro_auroc (mean per-epitope AUROC).
    """
    if "binding" not in df.columns or df["binding"].nunique() < 2:
        raise ValueError("provenance_floor requires a binary binding column")

    X_str = df["tcr"].values
    y     = df["binding"].values
    epis  = df["epi"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_scores, all_labels, all_epis, fold_aurocs = [], [], [], []

    for train_idx, test_idx in skf.split(X_str, y):
        vec = TfidfVectorizer(analyzer="char", ngram_range=ngram_range,
                              min_df=2, sublinear_tf=True)
        X_tr = vec.fit_transform(X_str[train_idx])
        X_te = vec.transform(X_str[test_idx])
        clf  = LogisticRegression(max_iter=2000, C=1.0, random_state=seed, solver="lbfgs")
        clf.fit(X_tr, y[train_idx])
        scores = clf.predict_proba(X_te)[:, 1]
        all_scores.append(scores)
        all_labels.append(y[test_idx])
        all_epis.append(epis[test_idx])
        if len(np.unique(y[test_idx])) > 1:
            fold_aurocs.append(float(roc_auc_score(y[test_idx], scores)))

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    all_epis   = np.concatenate(all_epis)

    pooled = float(roc_auc_score(all_labels, all_scores))

    macro_vals = []
    for epi in np.unique(all_epis):
        m = all_epis == epi
        if len(np.unique(all_labels[m])) < 2:
            continue
        try:
            macro_vals.append(float(roc_auc_score(all_labels[m], all_scores[m])))
        except Exception:
            pass

    return {
        "pooled_auroc": pooled,
        "macro_auroc":  float(np.mean(macro_vals))  if macro_vals else float("nan"),
        "macro_sd":     float(np.std(macro_vals))   if macro_vals else float("nan"),
        "macro_n":      len(macro_vals),
        "fold_aurocs":  fold_aurocs,
        "n_pairs":      int(len(all_labels)),
    }


# ── home-size channel probe ───────────────────────────────────────────────────

def home_size_channel(df):
    """
    Measure whether home_max (size of receptor's home epitope) leaks into
    within-epitope prediction.  Returns None for positives-only data.
    """
    if "binding" not in df.columns or df["binding"].nunique() < 2:
        return None

    epi_sizes = df.groupby("epi").size().to_dict()
    pos       = df[df["binding"] == 1]
    tcr_home  = (
        pos.groupby("tcr")["epi"]
        .apply(lambda ep: max(epi_sizes.get(e, 0) for e in ep))
        .to_dict()
    )

    df2           = df.copy()
    df2["home_max"] = df2["tcr"].map(tcr_home).fillna(0)
    y      = df2["binding"].values
    scores = df2["home_max"].values
    epis   = df2["epi"].values

    try:
        pooled = float(roc_auc_score(y, scores))
    except Exception:
        pooled = float("nan")

    macro_vals, gaps = [], []
    for epi in np.unique(epis):
        m = epis == epi
        if len(np.unique(y[m])) < 2:
            continue
        try:
            macro_vals.append(float(roc_auc_score(y[m], scores[m])))
        except Exception:
            pass
        pm = float(scores[m & (y == 1)].mean()) if (y[m] == 1).any() else float("nan")
        nm = float(scores[m & (y == 0)].mean()) if (y[m] == 0).any() else float("nan")
        if not np.isnan(pm) and not np.isnan(nm):
            gaps.append(pm - nm)

    top10 = sorted(epi_sizes.items(), key=lambda x: -x[1])[:10]
    return {
        "home_max_pooled":       pooled,
        "home_max_macro":        float(np.mean(macro_vals))  if macro_vals else float("nan"),
        "home_max_macro_sd":     float(np.std(macro_vals))   if macro_vals else float("nan"),
        "n_epitopes_evaluated":  len(macro_vals),
        "within_epi_gap_mean":   float(np.mean(gaps)) if gaps else None,
        "epi_size_distribution": {k: int(v) for k, v in top10},
    }


# ── positive-only dataset statistics ─────────────────────────────────────────

def positive_only_stats(df):
    epi_sizes = df.groupby("epi").size()
    return {
        "n_pairs":              int(len(df)),
        "n_epitopes":           int(df["epi"].nunique()),
        "n_tcrs":               int(df["tcr"].nunique()),
        "median_pairs_per_epi": float(epi_sizes.median()),
        "mean_pairs_per_epi":   float(epi_sizes.mean()),
        "max_pairs_per_epi":    int(epi_sizes.max()),
        "epitopes_with_1_pair": int((epi_sizes == 1).sum()),
        "top10_epitopes":       {k: int(v) for k, v in epi_sizes.nlargest(10).items()},
    }