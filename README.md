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
adata_cellsweep = cellsweep.denoise_count_matrix(adata_raw_path, adata_out=adata_cellsweep_path)  # assumes that adata_raw_path is an h5ad file or AnnData object with a column adata.obs['celltype'] indicating celltype

# for help
help(cellsweep.denoise_count_matrix)
```

### Command line interface
```
cellsweep denoise_count_matrix -o adata_cellsweep.h5ad adata_raw.h5ad  # assumes that adata_raw.h5ad is an h5ad file with a column adata.obs['celltype'] indicating celltype

# for help
cellsweep denoise_count_matrix --help
```

There are many utility functions in the `cellsweep.utils` module for data processing, plotting, and analysis. See examples in our Jupyter Notebooks.

## Tutorials
We have several Jupyter Notebooks demonstrating the use of CellSweep for denoising count matrices and analyzing the results. See the `notebooks` folder in the repository.