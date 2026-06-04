#!/usr/bin/env Rscript

# --- Parse command-line arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript run_soupx.R <matrix_tar_files_dir> <adata_obs_csv> <soupx_out_prefix> [cluster_col] [soup_range_max]\n")
  quit(status = 1)
}

matrix_tar_files_dir <- args[1]
adata_obs_csv        <- args[2]
soupx_out_prefix  <- args[3]
cluster_col          <- if (length(args) >= 4) args[4] else "leiden"
# Upper bound (UMIs) of the droplet range used to estimate the soup/background profile.
# SoupX's load10X default is (0, 100], but some datasets pre-filter low-count droplets so their
# empty droplets all sit above 100; pass the max empty-droplet UMI count here so the soup is
# estimated from the actual empty droplets. Defaults to 100 for backwards compatibility.
soup_range_max       <- if (length(args) >= 5) as.numeric(args[5]) else 100

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
# Build the SoupChannel manually rather than via load10X. SoupX's load10X -> SoupChannel calls
# estimateSoup() WITHOUT forwarding soupRange, so the soup is always estimated from droplets with
# UMIs in (0, 100]. Datasets that pre-filter low-count droplets have no droplets in that range,
# which yields an all-NaN soup profile and crashes autoEstCont. Constructing the channel with
# calcSoupProfile=FALSE and then calling estimateSoup() directly lets us pass the correct range.
locate_genome_subdir <- function(parent) {
  sub <- list.files(parent, full.names = TRUE)
  sub <- sub[dir.exists(sub)]
  if (length(sub) != 1) {
    stop(paste0("Expected exactly one genome subdirectory under ", parent, ", found ", length(sub)))
  }
  sub
}
raw_dir      <- locate_genome_subdir(file.path(matrix_tar_files_dir, "raw_gene_bc_matrices"))
filtered_dir <- locate_genome_subdir(file.path(matrix_tar_files_dir, "filtered_gene_bc_matrices"))

cat("Loading raw droplets from:", raw_dir, "\n")
tod <- Seurat::Read10X(raw_dir)  # Read10X lives in Seurat (SoupX imports it internally)
cat("Loading filtered cells from:", filtered_dir, "\n")
toc <- Seurat::Read10X(filtered_dir)
if (is.list(tod)) tod <- tod[[1]]
if (is.list(toc)) toc <- toc[[1]]

sc <- SoupChannel(tod, toc, calcSoupProfile = FALSE)
cat("Estimating soup profile from droplets with UMIs in (0,", soup_range_max, ")\n")
sc <- estimateSoup(sc, soupRange = c(0, soup_range_max))

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
