#!/usr/bin/env bash
# setup_external.sh -- download IMMREP23 and VDJdb raw data files
# Run from the repo root: bash setup_external.sh
set -euo pipefail
mkdir -p data results

# ── helpers ───────────────────────────────────────────────────────────────────
try_get() {
    local dest="$1"; shift
    for url in "$@"; do
        echo "  trying: $url"
        if wget -q --tries=2 --timeout=20 -O "$dest" "$url" 2>/dev/null; then
            local lines
            lines=$(wc -l < "$dest")
            if [ "$lines" -gt 1 ]; then
                echo "  ok ($lines lines) → $dest"
                return 0
            fi
        fi
        rm -f "$dest"
    done
    return 1
}

# ── VDJdb ─────────────────────────────────────────────────────────────────────
echo ""
echo "==> VDJdb"
if [ -f data/vdjdb_full.tsv ]; then
    echo "  data/vdjdb_full.tsv already exists ($(wc -l < data/vdjdb_full.tsv) lines)"
else
    # 1. Try latest GitHub release asset (slim TSV)
    ASSET_URL=$(wget -q -O - \
        "https://api.github.com/repos/antigenomics/vdjdb-db/releases/latest" \
        2>/dev/null \
        | python3 -c "
import json, sys
d = json.load(sys.stdin)
assets = [a['browser_download_url'] for a in d.get('assets', [])
          if a['name'].endswith('.txt') or a['name'].endswith('.tsv')
          or 'vdjdb' in a['name'].lower()]
print(assets[0] if assets else '')
" 2>/dev/null || true)

    DOWNLOADED=0

    if [ -n "$ASSET_URL" ]; then
        echo "  Latest release asset: $ASSET_URL"
        if wget -q --tries=2 --timeout=30 -O data/vdjdb_full.tsv "$ASSET_URL" 2>/dev/null; then
            DOWNLOADED=1
        elif wget -q --tries=2 --timeout=30 -O data/vdjdb_full.tsv.gz "$ASSET_URL" 2>/dev/null; then
            gunzip -f data/vdjdb_full.tsv.gz
            DOWNLOADED=1
        fi
    fi

    if [ "$DOWNLOADED" -eq 0 ]; then
        try_get data/vdjdb_full.tsv \
            "https://raw.githubusercontent.com/antigenomics/vdjdb-db/master/latest/vdjdb.txt" \
            "https://raw.githubusercontent.com/antigenomics/vdjdb-db/master/vdjdb.txt" \
        || {
            echo ""
            echo "  ✗ Automatic download failed."
            echo "  Download manually from: https://vdjdb.cdr3.ru"
            echo "  Click 'Browse' → Export → TSV → save as data/vdjdb_full.tsv"
            echo "  OR: go to https://github.com/antigenomics/vdjdb-db/releases"
            echo "  and download the .txt or .tsv asset, save as data/vdjdb_full.tsv"
            echo ""
        }
    fi
fi

# Peek at column headers if file exists
if [ -f data/vdjdb_full.tsv ]; then
    echo "  Columns: $(head -1 data/vdjdb_full.tsv)"
fi

# ── IMMREP23 ──────────────────────────────────────────────────────────────────
echo ""
echo "==> IMMREP23"
if [ -f data/immrep23_train.tsv ]; then
    echo "  data/immrep23_train.tsv already exists ($(wc -l < data/immrep23_train.tsv) lines)"
else
    try_get data/immrep23_train.tsv \
        "https://raw.githubusercontent.com/viragbioinfo/IMMREP/main/IMMREP_2023/Training/training_set.tsv" \
        "https://raw.githubusercontent.com/viragbioinfo/IMMREP/main/IMMREP_2023/training_set.tsv" \
        "https://raw.githubusercontent.com/viragbioinfo/IMMREP/main/IMMREP_2023/data/training_data.tsv" \
        "https://raw.githubusercontent.com/viragbioinfo/IMMREP/main/IMMREP23/TrainingData/IMMREP_2023_TrainingData.tsv" \
        "https://raw.githubusercontent.com/viragbioinfo/IMMREP/main/training_data/IMMREP_2023_train.tsv" \
    || {
        echo ""
        echo "  ✗ Automatic download failed."
        echo "  Try the IMMREP GitHub: https://github.com/viragbioinfo/IMMREP"
        echo "  Navigate to the 2023 challenge folder, download the training TSV,"
        echo "  and save it as: data/immrep23_train.tsv"
        echo ""
        echo "  If the paper is published, Supplementary Data may have it."
        echo "  Reference: ImmunoInformatics 2024, PMID check vignette."
        echo ""
    }
fi

if [ -f data/immrep23_train.tsv ]; then
    echo "  Columns: $(head -1 data/immrep23_train.tsv)"
fi

# ── canonicalise ──────────────────────────────────────────────────────────────
echo ""
echo "==> Canonicalising datasets"

HAVE_IMMREP=0
HAVE_VDJDB=0
[ -f data/immrep23_train.tsv ] && HAVE_IMMREP=1
[ -f data/vdjdb_full.tsv ]     && HAVE_VDJDB=1

if [ "$HAVE_IMMREP" -eq 0 ] && [ "$HAVE_VDJDB" -eq 0 ]; then
    echo "  No raw data files found. Download manually and re-run."
    exit 1
fi

export PYTHONPATH=src

python3 - << 'PY'
import sys, external as ex, pandas as pd

ref = ex.load_catelmo("data/TCREpitopePairs.csv")

datasets = []
import os

if os.path.exists("data/immrep23_train.tsv"):
    try:
        df = ex.load_immrep("data/immrep23_train.tsv")
        audit = ex.overlap_audit(df, ref)
        print(f"immrep23: {audit}")
        df.to_csv("data/immrep23_canon.csv", index=False)
        print("  → data/immrep23_canon.csv")
        datasets.append("immrep23")
    except Exception as e:
        print(f"  ERROR loading IMMREP23: {e}", file=sys.stderr)

if os.path.exists("data/vdjdb_full.tsv"):
    try:
        df = ex.load_vdjdb("data/vdjdb_full.tsv")
        audit = ex.overlap_audit(df, ref)
        print(f"vdjdb: {audit}")
        df.to_csv("data/vdjdb_canon.csv", index=False)
        print("  → data/vdjdb_canon.csv")
        datasets.append("vdjdb")
    except Exception as e:
        print(f"  ERROR loading VDJdb: {e}", file=sys.stderr)

print(f"\nCanonical files ready: {datasets}")
PY

echo ""
echo "==> Done. Next steps:"
echo ""
echo "  # IMMREP23 (has its own negatives — run battery directly)"
echo "  python3 src/run_external.py \\"
echo "      --csv data/immrep23_canon.csv --name immrep23 \\"
echo "      --ref data/TCREpitopePairs.csv --out results/ext_immrep23.json"
echo ""
echo "  # VDJdb (positives only — build negatives with our method first)"
echo "  python3 src/tcr_swap.py    --csv data/vdjdb_canon.csv --out data/vdjdb_swap.csv"
echo "  python3 src/fix_swap_pool.py --csv data/vdjdb_canon.csv --out data/vdjdb_matched.csv"
echo "  for c in swap matched; do"
echo "    python3 src/run_external.py \\"
echo "        --csv data/vdjdb_\${c}.csv --name vdjdb_\${c} \\"
echo "        --ref data/TCREpitopePairs.csv --out results/ext_vdjdb_\${c}.json"
echo "  done"