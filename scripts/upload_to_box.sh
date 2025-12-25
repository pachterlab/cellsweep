#!/usr/bin/env bash
set -euo pipefail

# Usage: ./upload_all.sh /path/to/hgmm12k
BASE="$1"
REMOTE_BASE="${2:-caltech_box:cellsweep}"   # default remote
CELLSWEEP_ONLY=false

# If a third argument is provided and matches flag, set it
if [[ "${3:-}" == "--cellsweep-only" ]]; then
    CELLSWEEP_ONLY=true
fi

# Extract dataset name (last component of path)
DATASET_NAME=$(basename "$BASE")

# Remote target
REMOTE="$REMOTE_BASE/$DATASET_NAME"

echo "Uploading dataset: $DATASET_NAME"
echo "Base directory: $BASE"
echo "Remote: $REMOTE"
echo "--------------------------------------------"

# --- Helper: check existence before running command ---
maybe_run() {
    local path="$1"
    shift
    if [[ -e "$path" ]]; then
        echo "[OK] Found: $path"
        "$@"
    else
        echo "[SKIP] Missing: $path"
    fi
}

# --- Helper: check all files exist before making tar ---
maybe_tar_and_copy() {
    local outfile="$1"
    shift
    local files=("$@")

    # Check all input files exist
    local all_exist=true
    for f in "${files[@]}"; do
        if [[ ! -e "$BASE/$f" ]]; then
            echo "[SKIP2] Missing: $BASE/$f — skipping tar $outfile"
            all_exist=false
        fi
    done

    if [[ "$all_exist" = true ]]; then
        echo "[OK] Creating tar: $outfile"
        tar -czvf "$BASE/$outfile" -C "$BASE" "${files[@]}"
        echo "[OK] Uploading: $outfile"
        rclone copy "$BASE/$outfile" "$REMOTE"
    fi
}

echo "============== Uploading =============="

# 1. Direct rclone uploads - cellsweep, CellBender, scAR
maybe_run "$BASE/${DATASET_NAME}_output_cellsweep.h5ad" \
    rclone copy "$BASE/${DATASET_NAME}_output_cellsweep.h5ad" "$REMOTE"

# If flag set → exit early
if [[ "$CELLSWEEP_ONLY" = true ]]; then
    echo "Cellsweep-only flag set. Exiting early."
    exit 0
fi

maybe_run "$BASE/${DATASET_NAME}_output_cellbender_filtered.h5" \
    rclone copy "$BASE/${DATASET_NAME}_output_cellbender_filtered.h5" "$REMOTE"

maybe_run "$BASE/cellbender_ckpt.tar.gz" \
    rclone copy "$BASE/cellbender_ckpt.tar.gz" "$REMOTE"

maybe_run "$BASE/${DATASET_NAME}_output_scar.h5" \
    rclone copy "$BASE/${DATASET_NAME}_output_scar.h5" "$REMOTE"

# 2. SoupX tar + upload
maybe_tar_and_copy \
    "${DATASET_NAME}_output_soupx.tar.gz" \
    "${DATASET_NAME}_output_soupx_barcodes.csv" \
    "${DATASET_NAME}_output_soupx_genes.csv" \
    "${DATASET_NAME}_output_soupx.mtx"

# 3. DecontX tar + upload
maybe_tar_and_copy \
    "${DATASET_NAME}_output_decontx.tar.gz" \
    "${DATASET_NAME}_output_decontx_barcodes.csv" \
    "${DATASET_NAME}_output_decontx_genes.csv" \
    "${DATASET_NAME}_output_decontx.mtx"

echo "--------------------------------------------"
echo "All uploads complete (missing files were skipped)."

