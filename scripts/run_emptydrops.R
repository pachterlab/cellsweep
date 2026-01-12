#!/usr/bin/env Rscript

# --- Parse command-line arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript run_emptydrops.R <raw_tar_file_dir> <out_file> [seed]\n")
  quit(status = 1)
}

raw_tar_file_dir      <- args[1]
out_file    <- args[2]

# Optional seed
if (length(args) >= 3) {
  seed <- as.integer(args[3])
  if (is.na(seed)) {
    stop("Seed must be an integer")
  }
} else {
  seed <- 123
}

set.seed(seed)

# --- Load libraries ---
suppressPackageStartupMessages({
  library(DropletUtils)
})

cat("📦 Starting EmptyDrops pipeline...\n")
cat("Raw matrix dir: ", raw_tar_file_dir, "\n")

sce <- read10xCounts(
    raw_tar_file_dir,
    col.names = TRUE
)

e.out <- emptyDrops(
    counts(sce),
    # lower = 100,        # droplets below this are assumed empty
    # niters = 10000,     # default; increase for large datasets
    test.ambient = TRUE
)

sce_filtered <- sce[, which(e.out$FDR <= 0.01)]
barcodes <- colnames(sce_filtered)

# ensure directory of out_file exists
dir.create(dirname(out_file), showWarnings = FALSE, recursive = TRUE)

# Write to text file (one barcode per line)
write.table(
  barcodes,
  file = out_file,
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)

cat("Filtered barcodes written to: ", out_file, "\n")

cat("✅ EmptyDrops completed successfully.\n")