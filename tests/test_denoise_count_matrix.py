import pytest
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from cellsweep import denoise_count_matrix
from cellsweep.model import infer_celltype_profile

# -----------------------
# Fixtures
# -----------------------
@pytest.fixture
def small_adata():
    """Create a small synthetic AnnData object with fake counts and celltypes."""
    X = np.array([
        [5, 0, 3],
        [0, 2, 1],
        [10, 1, 0],
        [0, 0, 0]  # empty droplet
    ])
    obs = pd.DataFrame({
        "celltype": ["A", "A", "B", "Empty Droplet"],
        "is_empty": [False, False, False, True]
    })
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    return ad.AnnData(X=X, obs=obs, var=var)


# -----------------------
# infer_celltype_profile
# -----------------------
def test_infer_celltype_profile(small_adata):
    adata = infer_celltype_profile(small_adata, celltype_key="celltype")
    assert "celltype_profile" in adata.uns
    assert "celltype_names" in adata.uns
    assert adata.uns["celltype_profile"].shape[1] == adata.n_vars
    assert len(adata.uns["celltype_names"]) > 0


# -----------------------
# denoise_count_matrix
# -----------------------
def test_denoise_count_matrix_runs(tmp_path, small_adata, monkeypatch):
    """Smoke test: ensure function runs and produces valid output."""

    # Mock expensive dependencies
    monkeypatch.setattr("cellsweep.utils.infer_empty_droplets", lambda *a, **kw: small_adata)
    monkeypatch.setattr("cellsweep.utils.load_adata", lambda a, logger=None: small_adata)

    out_path = tmp_path / "denoised.h5ad"

    adata_out = denoise_count_matrix(
        small_adata,
        adata_out=str(out_path),
        max_iter=2,
        verbose=-1,
        quiet=True
    )

    # Basic shape checks
    assert isinstance(adata_out, ad.AnnData)
    assert adata_out.X.shape == (small_adata.n_obs, small_adata.n_vars)

    # Check outputs were added
    assert "alpha_hat" in adata_out.obs
    assert "z_hat" in adata_out.obs
    assert "p_hat" in adata_out.uns

    # Check file written
    assert out_path.exists()

# -----------------------
# Edge cases
# -----------------------
def test_no_celltype_column_raises(small_adata):
    del small_adata.obs["celltype"]
    with pytest.raises(KeyError):
        infer_celltype_profile(small_adata)
