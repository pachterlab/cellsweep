# cellsweep

Sweep out noisy counts from single-cell RNA-seq data with CellSweep!

![alt text](https://github.com/pachterlab/cellsweep/blob/main/figures/logo.png?raw=true)

## Install
### Basic use
```
pip install cellsweep
```

### To run notebooks:
```
pip install cellsweep[analysis]
```

### To remake figures from the paper:
```
git clone https://github.com/pachterlab/cellsweep.git
cd cellsweep
conda env create -f environment.yml
pip install cellsweep[analysis]==0.1.1
```

## Quickstart
CellSweep has a single function denoise_count_matrix that takes a raw count matrix in an AnnData object and produces a denoised count matrix in another AnnData object. See a simple, fully worked example in the `notebooks/intro.ipynb` Jupyter Notebook.

### Python API
```python
import cellsweep
adata_cellsweep = cellsweep.denoise_count_matrix(adata_raw_path, adata_out=adata_cellsweep_path)  # see below for expected structure

# for help
help(cellsweep.denoise_count_matrix)
```

### Command line interface
```
cellsweep denoise_count_matrix -o adata_cellsweep.h5ad adata_raw.h5ad  # see below for expected structure

# for help
cellsweep denoise_count_matrix --help
```

There are many utility functions in the `cellsweep.utils` module for data processing, plotting, and analysis. See examples in our Jupyter Notebooks.

## Anndata object
The input Anndata object/h5ad file should have the following structure:
- `adata.X` : cell count matrix (cells x genes)
- `adata.obs`:
    - `adata.obs['celltype']`: a column indicating the cell type of each cell.
    - `adata.obs['is_empty']` (optional): a boolean column indicating whether each cell is an empty droplet. If not provided, CellSweep will infer empty droplets using the `empty_droplet_method` argument.
    - `adata.obs['init_alpha']` (optional): a column indicating the initial estimate of the fraction of ambient contamination for each cell. If not provided, CellSweep will use the `init_alpha` argument.
- `adata.var`:
    - `adata.var['ambient_profile']` (optional): a column indicating the per-gene ambient RNA fraction. If not provided, CellSweep will infer the ambient profile from the data.
- `adata.uns`:
    - `adata.uns['celltype_profile']` (optional): a matrix giving the mean expression for each cell type (K x G). If not provided, CellSweep will infer the cell type profile from the data.
    - `adata.uns['celltype_profile_genes']` (optional): a list of gene names corresponding to the columns of `celltype_profile`.

## Tutorials
We have several Jupyter Notebooks demonstrating the use of CellSweep for denoising count matrices and analyzing the results. See the `notebooks` folder in the repository.