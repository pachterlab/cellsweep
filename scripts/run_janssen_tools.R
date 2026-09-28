#!/usr/bin/env Rscript
# Reproduce the Janssen et al. (2023) default DecontX / SoupX runs on the mouse-kidney
# snRNA-seq replicates, reading the CellRanger raw h5 directly instead of their Seurat objects.
#
# Mirrors Snakemake_benchmark/scripts/run_DecontX.R and run_SoupX.R at the parameter
# settings flagged `default = TRUE` in their benchmark_metrics.RDS:
#   DecontX  resDefault_emptyFalse  and  resDefault_emptyTrue
#   SoupX    res1_contAuto
#
# Usage: Rscript run_janssen_tools.R <tool> <rep_dir> <out_prefix>
#   tool = decontx | decontx_empty | soupx

args <- commandArgs(trailingOnly = TRUE)
tool <- args[1]; rep_dir <- args[2]; out_prefix <- args[3]

suppressPackageStartupMessages({ library(Matrix) })

genes <- readLines(file.path(rep_dir, "seurat_genes.txt"))
cells <- readLines(file.path(rep_dir, "seurat_cells.txt"))
md    <- read.csv(file.path(rep_dir, "seurat_metadata.csv"), check.names = FALSE)
rownames(md) <- md$cell

cat("Reading raw CellRanger h5...\n")
sce <- DropletUtils::read10xCounts(file.path(rep_dir, "raw_feature_bc_matrix.h5"),
                                   col.names = TRUE)
tod <- SingleCellExperiment::counts(sce)
rownames(tod) <- make.unique(as.character(SummarizedExperiment::rowData(sce)$Symbol))
rm(sce); gc()
cat("  raw:", nrow(tod), "genes x", ncol(tod), "barcodes\n")
stopifnot(all(genes %in% rownames(tod)), all(cells %in% colnames(tod)))

write_out <- function(m, prefix) {
  cat("Writing", prefix, "\n")
  Matrix::writeMM(as(m, "dgCMatrix"), file = paste0(prefix, ".mtx"))
  write.table(rownames(m), paste0(prefix, "_genes.csv"), row.names = FALSE,
              col.names = FALSE, quote = FALSE)
  write.table(colnames(m), paste0(prefix, "_barcodes.csv"), row.names = FALSE,
              col.names = FALSE, quote = FALSE)
}

if (tool %in% c("decontx", "decontx_empty")) {
  suppressPackageStartupMessages(library(celda))
  use_empty <- identical(tool, "decontx_empty")
  counts <- tod[genes, cells]
  background <- if (use_empty) tod[genes, ] else NULL
  cat("Running decontX (empty droplets:", use_empty, ", z = NULL)\n")
  res <- decontX(x = as.matrix(counts), z = NULL, background = background)
  out <- res$decontXcounts
  write.csv(data.frame(cell = colnames(counts), cont = res$contamination),
            paste0(out_prefix, "_contPerCell.csv"), row.names = FALSE)
  write_out(out, out_prefix)
} else if (tool == "soupx") {
  suppressPackageStartupMessages(library(SoupX))
  # run_SoupX.R builds the channel from the full CellRanger matrix with load10X(), which
  # estimates the soup from droplets holding 1-99 UMIs, and only then drops the features that
  # are missing from the Seurat object -- without renormalising the profile or recomputing the
  # per-cell UMI totals. Both details matter here, because more than half of the soup sits in
  # the 13 mitochondrial genes that get dropped, so we reproduce the order exactly. The one
  # departure is that the soup row sums are precomputed (soup_profile.csv, see the notebook)
  # rather than obtained by subsetting a 6.8M-column matrix inside R, which is intractable.
  soup <- read.csv(file.path(rep_dir, "soup_profile.csv"))
  cat("  soup profile:", sum(soup$counts), "counts;",
      round(100 * sum(soup$counts[soup$gene %in% genes]) / sum(soup$counts), 1),
      "% outside the mitochondrial genes\n")

  # read10xCounts() hands back an HDF5-backed DelayedMatrix; SoupX needs a real dgCMatrix.
  all_genes <- rownames(tod)
  toc <- Matrix::Matrix(as.matrix(tod[, cells]), sparse = TRUE)
  rm(tod); gc()
  sc <- SoupChannel(toc, toc, calcSoupProfile = FALSE)
  counts_all <- setNames(soup$counts, soup$gene)[all_genes]
  sc$soupProfile <- data.frame(est = counts_all / sum(counts_all), counts = counts_all,
                               row.names = all_genes)

  cl <- as.character(md[cells, "RNA_snn_res.1"]); names(cl) <- cells
  sc <- setClusters(sc, cl)
  sc$toc <- sc$toc[genes, ]
  sc$soupProfile <- sc$soupProfile[genes, ]
  cat("Estimating contamination fraction (autoEstCont)...\n")
  sc <- autoEstCont(sc)
  cat("Adjusting counts...\n")
  out <- adjustCounts(sc)
  write.csv(data.frame(cell = colnames(out),
                       cont = 1 - (Matrix::colSums(out) / Matrix::colSums(sc$toc))),
            paste0(out_prefix, "_contPerCell.csv"), row.names = FALSE)
  write_out(out, out_prefix)
} else {
  stop("unknown tool: ", tool)
}
cat("done\n")
