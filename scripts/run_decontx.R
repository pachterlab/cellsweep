#!/usr/bin/env Rscript

# --- Parse command-line arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  cat("Usage: Rscript run_decontx.R <raw_tar_file_dir> <filtered_tar_file_dir> <sequencing_technology> <decontx_out_prefix> [--prepend_sample_to_barcodes] [--clusters_csv=PATH] [--cluster_col=COL]\n")
  quit(status = 1)
}

raw_tar_file_dir      <- args[1]
filtered_tar_file_dir <- args[2]
sequencing_technology <- toupper(args[3])
decontx_out_prefix    <- args[4]
prepend_sample_to_barcodes <- ifelse("--prepend_sample_to_barcodes" %in% args, TRUE, FALSE)

# --- Optional cluster labels for decontX (z) ---
# By default decontX performs its own internal clustering. To benchmark sensitivity to the
# cell-typing / clustering method, pass externally computed cluster labels (e.g. Leiden at a
# chosen resolution) via --clusters_csv (a CSV indexed by barcode) and --cluster_col (the column
# holding the labels, defaults to "leiden"). When provided, the labels are forwarded to decontX
# as z, replacing its native clustering. Barcodes are matched after stripping any genome prefix
# (e.g. "GRCh38_") that importCellRanger prepends to the SCE column names.
get_kv_arg <- function(flag, default = NULL) {
  hit <- grep(paste0("^", flag, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^", flag, "="), "", hit[1])
}
clusters_csv <- get_kv_arg("--clusters_csv", NULL)
cluster_col  <- get_kv_arg("--cluster_col", "leiden")

# --- Load libraries ---
suppressPackageStartupMessages({
  library(singleCellTK)
  library(decontX)
  library(Matrix)
})

cat("📦 Starting DecontX pipeline...\n")
cat("Raw matrix dir: ", raw_tar_file_dir, "\n")
cat("Filtered matrix dir: ", filtered_tar_file_dir, "\n")
cat("Sequencing technology: ", sequencing_technology, "\n")

# --- Helper to import Cell Ranger data based on technology ---
load_cellranger <- function(data_dir, data_type = c("filtered", "raw"), technology) {
  data_type <- match.arg(data_type)
  cat(sprintf("Importing %s CellRanger matrix for %s...\n", data_type, technology))

  if (technology == "10XV1") {
      #!!! needs debugging
      fn <- importCellRanger
      if (data_type == "raw") {
          return(fn(sampleDirs = data_dir, dataType = "raw"))
      } else {
          return(fn(sampleDirs = data_dir))
      }
  } else if (technology == "10XV2") {
      fn <- importCellRangerV2Sample
      return(fn(dataDir = data_dir))
  } else if (technology == "10XV3") {
      fn <- importCellRangerV3Sample
      return(fn(dataDir = data_dir))
  } else {
      stop(paste0("❌ Unsupported sequencing technology: ", technology,
                  ". Must be one of: 10XV1, 10XV2, 10XV3."))
  }
}

# --- Load filtered and raw data ---
sce      <- load_cellranger(filtered_tar_file_dir, data_type = "filtered", technology = sequencing_technology)
sce.raw  <- load_cellranger(raw_tar_file_dir,      data_type = "raw",      technology = sequencing_technology)

# --- Standardize column and row names ---
cat("Standardizing cell and gene names...\n")
if (prepend_sample_to_barcodes) {
  colnames(sce) <- paste(sce$sample, sce$cell_barcode, sep = "_")   #!!! might need to modify this line based on the dataset (if cell csv is empty after running)
} else {
  colnames(sce) <- sce$cell_barcode
}
rownames(sce) <- rowData(sce)$feature_name   #!!! might need to modify this line based on the dataset (if gene csv is empty after running)
counts(sce) <- as(counts(sce), "dgCMatrix")

# --- Optionally build cluster labels (z) from an external CSV ---
z_labels <- NULL
if (!is.null(clusters_csv)) {
  cat("Loading external cluster labels from:", clusters_csv, "(column:", cluster_col, ")\n")
  clusters_df <- read.csv(clusters_csv, row.names = 1, check.names = FALSE)
  if (!cluster_col %in% colnames(clusters_df)) {
    stop(paste0("Column '", cluster_col, "' not found in clusters_csv."))
  }
  clu <- as.character(clusters_df[[cluster_col]])
  names(clu) <- rownames(clusters_df)

  # Match against the (possibly genome-prefixed) SCE column names. Try a direct match first,
  # then fall back to the bare barcode after stripping a leading "<genome>_" prefix.
  cn <- colnames(sce)
  bare <- sub("^[^_]+_", "", cn)
  z_labels <- clu[cn]
  z_labels[is.na(z_labels)] <- clu[bare][is.na(z_labels)]

  n_missing <- sum(is.na(z_labels))
  if (n_missing > 0) {
    stop(paste0(n_missing, " of ", length(cn),
                " SCE barcodes had no matching cluster label in clusters_csv."))
  }
  z_labels <- as.character(z_labels)
  cat("Assigned", length(unique(z_labels)), "unique clusters across", length(z_labels), "cells.\n")
}

# --- Run decontX ---
cat("Running decontX denoising...\n")
if (is.null(z_labels)) {
  sce <- decontX(sce, background = sce.raw)
} else {
  sce <- decontX(sce, z = z_labels, background = sce.raw)
}

# --- Extract and write outputs ---
cat("Writing corrected count matrix...\n")
decontx_counts <- assay(sce, "decontXcounts")

cat("Writing output matrices...\n")
Matrix::writeMM(decontx_counts, file = paste0(decontx_out_prefix, ".mtx"))
write.table(rownames(decontx_counts), file = paste0(decontx_out_prefix, "_genes.csv"), row.names = FALSE, col.names = FALSE, quote = FALSE, sep = ",")
write.table(colnames(decontx_counts), file = paste0(decontx_out_prefix, "_barcodes.csv"), row.names = FALSE, col.names = FALSE, quote = FALSE, sep = ",")

cat("✅ DecontX completed successfully.\n")