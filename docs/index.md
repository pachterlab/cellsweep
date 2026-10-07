# cellsweep

```{image} https://github.com/pachterlab/cellsweep/blob/main/figures/logo.png?raw=true
:alt: cellsweep logo
:width: 300px
:align: center
```

Sweep out noisy counts from single-cell RNA-seq data with **CellSweep**.

CellSweep removes ambient RNA and global (bulk) contamination from droplet-based
single-cell count matrices. It fits a multinomial mixture model with an
expectation–maximization (EM) algorithm that assigns each observed count to one of three
sources: the cell's true cell-type expression, ambient contamination, or bulk
contamination. The output is a denoised count matrix in an
[AnnData](https://anndata.readthedocs.io) object.

```{tip}
New here? Start with {doc}`installation` and {doc}`quickstart`, then work through the
{doc}`tutorials/intro` tutorial.
```

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
input_format
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/intro
tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

cli
api/index
advanced_parameters
```

```{toctree}
:maxdepth: 1
:caption: Project

GitHub <https://github.com/pachterlab/cellsweep>
PyPI <https://pypi.org/project/cellsweep/>
```
