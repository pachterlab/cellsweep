#!/usr/bin/env Rscript

# --- Parse command-line arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript run_soupx.R <matrix_tar_files_dir> <adata_obs_csv> <anndata_out>\n")
  quit(status = 1)
}

matrix_tar_files_dir <- args[1]
adata_obs_csv        <- args[2]
anndata_out          <- args[3]

# --- Load libraries ---
suppressPackageStartupMessages({
  library(SoupX)
  library(Matrix)
  library(anndata)
})

# --- Load obs (clusters) ---
adata_soupx_tmp_obs <- read.csv(adata_obs_csv, row.names = 1)
clusters <- adata_soupx_tmp_obs$leiden
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
# obs (cells) and var (genes) as data frames with proper row names
obs_df <- data.frame(cell_id = colnames(out), row.names = colnames(out))
var_df <- data.frame(gene_id = rownames(out), row.names = rownames(out))

ad <- AnnData(
  X = out,
  obs = obs_df,
  var = var_df,
  uns = list(
    metadata = list(
      method = "SoupX",
      date = Sys.time()
    )
  )
)

write_h5ad(ad, anndata_out)

cat("✅ SoupX completed successfully.\n")
