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
# Optional override for the per-batch cell cap used when adjusting counts (see below). When <= 0
# (the default), the cap is derived automatically from the matrix non-zero count.
max_cells_per_batch_arg <- if (length(args) >= 6) as.numeric(args[6]) else 0

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
# R's Matrix package cannot build a TsparseMatrix whose i/j index vectors exceed 2^31-1, so
# adjustCounts() overflows on very large matrices (the soup subtraction concatenates two index
# vectors, ~2x the non-zero count). We therefore run adjustCounts() in batches of WHOLE clusters
# and recombine downstream in Python. This is exact: SoupX's cluster-level adjustment and
# expandClusters() both operate per-cluster independently, so grouping whole clusters per batch
# yields identical per-cell results to a single call. Small datasets stay a single batch/file so
# the original (single soupx_out.mtx) output format is preserved for backwards compatibility.
cell_ids <- colnames(sc$toc)
clu <- as.character(clusters[cell_ids])
names(clu) <- cell_ids

nnz_total <- length(sc$toc@x)
avg_nnz_per_cell <- nnz_total / length(cell_ids)
# Cap per-batch non-zeros at 5e8 so the ~2x concatenation stays well under the 2^31-1 limit.
if (max_cells_per_batch_arg > 0) {
  max_cells_per_batch <- as.integer(max_cells_per_batch_arg)
} else {
  max_cells_per_batch <- max(1, floor(5e8 / max(1, avg_nnz_per_cell)))
}

batches <- list()
cur <- character(0)
for (cl in unique(clu)) {
  cells_cl <- cell_ids[clu == cl]
  if (length(cur) > 0 && (length(cur) + length(cells_cl)) > max_cells_per_batch) {
    batches[[length(batches) + 1]] <- cur
    cur <- character(0)
  }
  cur <- c(cur, cells_cl)
}
if (length(cur) > 0) batches[[length(batches) + 1]] <- cur
cat(sprintf("Adjusting counts in %d batch(es) of whole clusters (max ~%d cells/batch)...\n",
            length(batches), max_cells_per_batch))

genes_out <- rownames(sc$toc)
write.table(genes_out, file = paste0(soupx_out_prefix, "_genes.csv"), row.names = FALSE, col.names = FALSE, quote = FALSE, sep = ",")

if (length(batches) == 1) {
  out <- adjustCounts(sc)
  cat("Writing output matrices...\n")
  Matrix::writeMM(out, file = paste0(soupx_out_prefix, ".mtx"))
  write.table(colnames(out), file = paste0(soupx_out_prefix, "_barcodes.csv"), row.names = FALSE, col.names = FALSE, quote = FALSE, sep = ",")
} else {
  for (b in seq_along(batches)) {
    cat(sprintf("  Adjusting batch %d/%d (%d cells)...\n", b, length(batches), length(batches[[b]])))
    sc_b <- sc
    sc_b$toc <- sc$toc[, batches[[b]], drop = FALSE]
    sc_b$metaData <- sc$metaData[batches[[b]], , drop = FALSE]
    out_b <- adjustCounts(sc_b)
    Matrix::writeMM(out_b, file = paste0(soupx_out_prefix, "_batch", b, ".mtx"))
    write.table(colnames(out_b), file = paste0(soupx_out_prefix, "_batch", b, "_barcodes.csv"), row.names = FALSE, col.names = FALSE, quote = FALSE, sep = ",")
    rm(out_b, sc_b); gc()
  }
  writeLines(as.character(length(batches)), con = paste0(soupx_out_prefix, "_nbatches.txt"))
}

cat("✅ SoupX completed successfully.\n")
