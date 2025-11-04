#!/usr/bin/env Rscript

# --- Parse command-line arguments ---
if (length(args) < 4) {
  cat("Usage: Rscript run_decontx.R <raw_tar_file_dir> <filtered_tar_file_dir> <sequencing_technology> <decontx_out_prefix>\n")
  quit(status = 1)
}

raw_tar_file_dir      <- args[1]
filtered_tar_file_dir <- args[2]
sequencing_technology <- toupper(args[3])
decontx_out_prefix    <- args[4]

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
colnames(sce) <- paste(sce$sample, sce$cell_barcode, sep = "_")
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