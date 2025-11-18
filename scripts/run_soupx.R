#!/usr/bin/env Rscript

# --- Parse command-line arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript run_soupx.R <matrix_tar_files_dir> <adata_obs_csv> <soupx_out_prefix> [cluster_col]\n")
  quit(status = 1)
}

matrix_tar_files_dir <- args[1]
adata_obs_csv        <- args[2]
soupx_out_prefix  <- args[3]
cluster_col          <- if (length(args) >= 4) args[4] else "leiden"

# --- Load libraries ---
suppressPackageStartupMessages({
  library(SoupX)
  library(Matrix)
})

# --- Load obs (clusters) ---
adata_soupx_tmp_obs <- read.csv(adata_obs_csv, row.names = 1)
if (!cluster_col %in% colnames(adata_soupx_tmp_obs)) {
  stop(paste0("Column '", cluster_col, "' not found in adata_obs_csv."))
}
clusters <- adata_soupx_tmp_obs[[cluster_col]]
names(clusters) <- rownames(adata_soupx_tmp_obs)

# --- Run SoupX ---
cat("Loading data from:", matrix_tar_files_dir, "\n")
sc <- load10X(matrix_tar_files_dir)

cat("Assigning clusters...\n")
sc <- setClusters(sc, clusters)

cat("Estimating contamination fraction...\n")
sc <- autoEstCont(sc)

cat("Adjusting counts...\n")
out <- adjustCounts(sc)

cat("Writing output matrices...\n")
Matrix::writeMM(out, file = paste0(soupx_out_prefix, ".mtx"))
write.table(rownames(out), file = paste0(soupx_out_prefix, "_genes.csv"), row.names = FALSE, col.names = FALSE, quote = FALSE, sep = ",")
write.table(colnames(out), file = paste0(soupx_out_prefix, "_barcodes.csv"), row.names = FALSE, col.names = FALSE, quote = FALSE, sep = ",")


cat("✅ SoupX completed successfully.\n")
