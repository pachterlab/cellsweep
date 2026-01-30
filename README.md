# cellsweep

Sweep out noisy counts from single-cell RNA-seq data with CellSweep!

![alt text](https://github.com/pachterlab/cellsweep/blob/main/figures/logo.png?raw=true)

## Install
### Basic use
```
pip install cellsweep  #!!! not yet uploaded on pypi - see below
```

### To run notebooks:
```
pip install cellsweep[analysis]  # pip install cellsweep["analysis"] for Mac users  #!!! not yet uploaded on pypi - see below
```

### To remake figures from the paper:
```
git clone https://github.com/pachterlab/cellsweep.git
cd cellsweep
conda env create -f environment.yml  #!!! until on pypi, in environment.yml replace cellsweep[analysis]==0.1.0 with .[analysis]
```

## Quickstart
CellSweep has a single function denoise_count_matrix that takes a raw count matrix in an AnnData object and produces a denoised count matrix in another AnnData object. See a simple, fully worked example in the `notebooks/intro.ipynb` Jupyter Notebook.

### Python API
```python
from cellsweep import denoise_count_matrix
adata_cellsweep = denoise_count_matrix(adata_raw_path, adata_out=adata_cellsweep_path)  # assumes that adata_raw_path is an h5ad file or AnnData object with a column adata.obs['celltype'] indicating celltype
```

### Command line interface
```
cellsweep denoise_count_matrix -o adata_cellsweep.h5ad adata_raw.h5ad  # assumes that adata_raw.h5ad is an h5ad file with a column adata.obs['celltype'] indicating celltype
```

There are many utility functions in the `cellsweep.utils` module for data processing, plotting, and analysis. See examples in our Jupyter Notebooks.

## Tutorials
We have several Jupyter Notebooks demonstrating the use of CellSweep for denoising count matrices and analyzing the results. See the `notebooks` folder in the repository.