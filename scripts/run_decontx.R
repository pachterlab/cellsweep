#!/usr/bin/env Rscript

# --- Parse command-line arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript run_decontx.R <raw_tar_file_dir> <filtered_tar_file_dir> <anndata_out>\n")
  quit(status = 1)
}

raw_tar_file_dir      <- args[1]
filtered_tar_file_dir <- args[2]
anndata_out <- args[3]

# --- Load libraries ---
suppressPackageStartupMessages({
  library(SingleCellTK)
  library(decontX)
  library(Matrix)
  library(anndata)
})

cat("📦 Starting DecontX pipeline...\n")
cat("Raw matrix dir: ", raw_tar_file_dir, "\n")
cat("Filtered matrix dir: ", filtered_tar_file_dir, "\n")

# --- Load filtered data (cells) ---
cat("Importing filtered CellRanger matrix...\n")
sce <- importCellRanger(sampleDirs = filtered_tar_file_dir)

# --- Load raw data (ambient background) ---
cat("Importing raw CellRanger matrix...\n")
sce.raw <- importCellRanger(sampleDirs = raw_tar_file_dir, dataType = "raw")

# --- Standardize column and row names ---
cat("Standardizing cell and gene names...\n")
colnames(sce) <- paste(sce$Sample, sce$Barcode, sep = "_")
rownames(sce) <- rowData(sce)$Symbol_TENx
counts(sce) <- as(counts(sce), "dgCMatrix")

# --- Run decontX ---
cat("Running decontX denoising...\n")
sce <- decontX(sce, background = sce.raw)

# --- Extract and write outputs ---
cat("Writing corrected count matrix...\n")
decontx_counts <- assay(sce, "decontXcounts")

# Create obs / var data frames
obs_df <- data.frame(cell_id = colnames(decontx_counts), row.names = colnames(decontx_counts))
var_df <- data.frame(gene_id = rownames(decontx_counts), row.names = rownames(decontx_counts))

ad <- AnnData(
  X = decontx_counts,
  obs = obs_df,
  var = var_df,
  uns = list(
    metadata = list(
      method = "decontX",
      date = Sys.time()
    )
  )
)

write_h5ad(ad, anndata_out)

cat("✅ DecontX completed successfully.\n")