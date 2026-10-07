# Installation

CellSweep requires Python 3.9 or later.

## Basic use

```bash
pip install cellsweep
```

This installs the core package (NumPy, Numba, pandas, SciPy, AnnData, pydantic), which is
all you need to run {func}`cellsweep.denoise` and the `cellsweep` command line tool.

## Running the notebooks

The tutorial notebooks and the plotting helpers in {mod}`cellsweep.utils` use extra
packages (scanpy, matplotlib, seaborn, CellTypist, ...). Install them with:

```bash
pip install "cellsweep[analysis]"
```

## Remaking the figures from the paper

```bash
git clone https://github.com/pachterlab/cellsweep.git
cd cellsweep
conda env create -f environment.yml
pip install "cellsweep[analysis]==0.1.1"
```

## Development install

```bash
git clone https://github.com/pachterlab/cellsweep.git
cd cellsweep
pip install -e ".[dev]"
pytest
```

## Building these docs locally

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.
