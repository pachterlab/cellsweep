"""Sphinx configuration for the cellsweep documentation."""

import os
import shutil
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import cellsweep  # noqa: E402

# -- Project information -----------------------------------------------------

project = "cellsweep"
author = "Maya Caskey, Joseph Rich"
copyright = "2025, Maya Caskey, Joseph Rich"
release = cellsweep.__version__
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxarg.ext",
    "myst_nb",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Notebooks are rendered with their saved outputs; they are never executed on Read the Docs.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# Notebooks live in the repo's notebooks/ folder; copy the ones we render into the docs tree.
TUTORIAL_NOTEBOOKS = ["intro.ipynb"]
for _nb in TUTORIAL_NOTEBOOKS:
    _src = REPO_ROOT / "notebooks" / _nb
    if _src.exists():
        shutil.copy2(_src, DOCS_DIR / "tutorials" / _nb)

# -- Autodoc -----------------------------------------------------------------

autosummary_generate = True
autodoc_typehints = "none"  # the numpy-style docstrings already document types
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True}

# Optional plotting / analysis dependencies that visualization_utils imports.
autodoc_mock_imports = [
    "matplotlib",
    "seaborn",
    "scanpy",
    "upsetplot",
    "celltypist",
    "mpl_scatter_density",
    "astropy",
    "sklearn",
    "torch",
    "adjustText",
    "squidpy",
    "tqdm",
    "requests",
    "yaml",
    "mpl_toolkits",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"cellsweep {release}"
html_logo = str(REPO_ROOT / "figures" / "logo.png")
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/pachterlab/cellsweep",
    "source_branch": "main",
    "source_directory": "docs/",
}
