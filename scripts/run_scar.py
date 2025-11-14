import os
import scanpy as sc
from scar import model, setup_anndata
import warnings
import argparse
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Run scAR on 20k HGMM dataset')
parser.add_argument('-r', '--raw', type=str, required=True, help='Path to raw count matrix in h5ad format or directory to .mtx/barcodes/genes files')
parser.add_argument('-f', '--filtered', type=str, required=True, help='Path to filtered count matrix in h5ad format or directory to .mtx/barcodes/genes files')
parser.add_argument('-o', '--output', type=str, default="adata_scar_denoised.h5ad", help='Path to output file')
parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--min_counts', type=int, default=200, help='Minimum counts per gene to retain')
parser.add_argument('--max_counts', type=int, default=6000, help='Maximum counts per gene to retain')
parser.add_argument('--min_genes', type=int, default=0, help='Minimum number of genes per cell to retain')
args = parser.parse_args()

print("Loading data...")
if args.raw.endswith('.h5'):
    adata_raw = sc.read_10x_h5(filename=args.raw)
elif os.path.isdir(args.raw):
    adata_raw = sc.read_10x_mtx(args.raw)
else:
    raise ValueError("Invalid path for raw count matrix. Provide either an h5 file or a directory containing mtx/barcodes/genes files.")
adata_raw.var_names_make_unique()
adata_raw.var["feature_types"] = "Gene Expression"

if args.filtered.endswith('.h5'):
    adata = sc.read_10x_h5(filename=args.filtered)
elif os.path.isdir(args.filtered):
    adata = sc.read_10x_mtx(args.filtered)
else:
    raise ValueError("Invalid path for filtered count matrix. Provide either an h5 file or a directory containing mtx/barcodes/genes files.")
adata.var_names_make_unique()
adata.var["feature_types"] = "Gene Expression"

print("Preprocessing data...")
if args.min_counts:
    sc.pp.filter_genes(adata, min_counts=args.min_counts)
if args.max_counts:
    sc.pp.filter_genes(adata, max_counts=args.max_counts)
if args.min_genes:
    sc.pp.filter_cells(adata, min_genes=args.min_genes)

print("Setting up AnnData for scAR...")
setup_anndata(
    adata = adata,
    raw_adata = adata_raw,
    feature_type = "Gene Expression",
    prob = 0.995,
    kneeplot = True
)

print("Running scAR...")
device = 'cuda' if args.cuda else 'cpu'
adata_scar = model(raw_count=adata, # In the case of Anndata object, scar will automatically use the estimated ambient_profile present in adata.uns.
                      ambient_profile=adata.uns['ambient_profile_Gene Expression'],
                      feature_type='mRNA',
                      sparsity=1,
                      device=device # CPU, CUDA and MPS are supported.
                     )

adata_scar.train(epochs=args.epochs,
                    batch_size=64,
                    verbose=True
                   )

print("Performing inference...")
adata_scar.inference()  # by defaut, batch_size = None, set a batch_size if getting a memory issue
assert adata_scar.native_counts.X.shape == adata.X.shape, "Denoised count matrix shape does not match input count matrix shape."

print("Saving results...")
adata.layers["raw"] = adata.X.copy()  # save raw counts in layers
adata.X = adata_scar.native_counts

if os.path.dirname(args.output):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
adata.write_h5ad(args.output)