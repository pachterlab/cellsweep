#!/usr/bin/env Rscript

# --- Parse command-line arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript run_decontx.R <raw_tar_file_dir> <filtered_tar_file_dir> <decontx_out_prefix>\n")
  quit(status = 1)
}

raw_tar_file_dir      <- args[1]
filtered_tar_file_dir <- args[2]
decontx_out_prefix    <- args[3]

# --- Load libraries ---
suppressPackageStartupMessages({
  library(SingleCellTK)
  library(decontX)
  library(Matrix)
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

cat("Writing output matrices...\n")
Matrix::writeMM(decontx_counts, file = paste0(decontx_out_prefix, ".mtx"))
write.table(rownames(decontx_counts), file = paste0(decontx_out_prefix, "_genes.csv"), row.names = FALSE, col.names = FALSE, quote = FALSE, sep = ",")
write.table(colnames(decontx_counts), file = paste0(decontx_out_prefix, "_barcodes.csv"), row.names = FALSE, col.names = FALSE, quote = FALSE, sep = ",")

cat("✅ DecontX completed successfully.\n")