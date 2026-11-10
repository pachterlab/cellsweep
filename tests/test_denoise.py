import pytest
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import sys
from cellsweep import denoise
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


def test_infer_celltype_profile_ignores_labels_on_empties(small_adata):
    """A real cell-type label on an empty droplet does not change that cell type's profile."""
    labeled = small_adata.copy()
    labeled.obs["celltype"] = ["A", "A", "B", "A"]
    ref = infer_celltype_profile(small_adata.copy(), celltype_key="celltype")
    out = infer_celltype_profile(labeled, celltype_key="celltype")
    np.testing.assert_allclose(out.uns["celltype_profile"], ref.uns["celltype_profile"])


# -----------------------
# empty_label
# -----------------------
@pytest.fixture
def no_flag_adata(small_adata):
    """No is_empty column, and the labeled empty droplet has enough counts to pass a UMI cutoff of 1."""
    adata = small_adata.copy()
    del adata.obs["is_empty"]
    adata.X[3] = [1, 1, 1]
    return adata


def test_empty_label_marks_empties(no_flag_adata):
    out = denoise(no_flag_adata, keep_empties=True, umi_cutoff=1, max_iter=2, verbose=-1, quiet=True)
    assert out.obs["is_empty"].tolist() == [False, False, False, True]
    assert "Empty Droplet" not in out.uns["celltype_names"]


def test_empty_label_custom(no_flag_adata):
    no_flag_adata.obs["celltype"] = ["A", "A", "B", "blank"]
    out = denoise(no_flag_adata, keep_empties=True, empty_label="blank", umi_cutoff=1, max_iter=2, verbose=-1, quiet=True)
    assert out.obs["is_empty"].tolist() == [False, False, False, True]


def test_empty_label_none_uses_umi_inference(no_flag_adata):
    out = denoise(no_flag_adata, keep_empties=True, empty_label=None, umi_cutoff=1, max_iter=2, verbose=-1, quiet=True)
    assert not out.obs["is_empty"].any()
    assert "Empty Droplet" in out.uns["celltype_names"]


def test_empty_label_missing_labels_are_not_empty(no_flag_adata):
    """A missing label can be an unannotated cell, so it is never read as empty."""
    no_flag_adata.obs["celltype"] = pd.Categorical(["A", None, "B", "Empty Droplet"])
    out = denoise(no_flag_adata, keep_empties=True, umi_cutoff=1, max_iter=2, verbose=-1, quiet=True)
    assert out.obs["is_empty"].tolist() == [False, False, False, True]


@pytest.mark.parametrize("label", [None, "A", "only_on_empties"])
def test_denoise_ignores_labels_on_empties(small_adata, label):
    """When is_empty is True, the barcode's label has no effect on the output."""
    small_adata.X[3] = [1, 1, 1]
    ref = denoise(small_adata.copy(), keep_empties=True, max_iter=2, verbose=-1, quiet=True)
    relabeled = small_adata.copy()
    relabeled.obs["celltype"] = pd.Categorical(["A", "A", "B", label])
    out = denoise(relabeled, keep_empties=True, max_iter=2, verbose=-1, quiet=True)

    assert list(out.uns["celltype_names"]) == ["A", "B"]
    np.testing.assert_array_equal(out.uns["p_hat"], ref.uns["p_hat"])
    np.testing.assert_array_equal(out.obs["z_hat"], ref.obs["z_hat"])
    np.testing.assert_array_equal(out.X.toarray() if sp.issparse(out.X) else out.X,
                                  ref.X.toarray() if sp.issparse(ref.X) else ref.X)


def test_is_empty_column_takes_priority_over_label(small_adata):
    small_adata.obs["is_empty"] = [False, False, True, True]
    out = denoise(small_adata, keep_empties=True, max_iter=2, verbose=-1, quiet=True)
    assert out.obs["is_empty"].tolist() == [False, False, True, True]


# -----------------------
# denoise
# -----------------------
def test_denoise_runs(tmp_path, small_adata, monkeypatch):
    """Smoke test: ensure function runs and produces valid output."""

    # Mock expensive dependencies
    monkeypatch.setattr("cellsweep.utils.infer_empty_droplets", lambda *a, **kw: small_adata)
    monkeypatch.setattr("cellsweep.utils.load_adata", lambda a, logger=None: small_adata)

    out_path = tmp_path / "denoised.h5ad"

    adata_out = denoise(
        small_adata,
        adata_out=str(out_path),
        max_iter=2,
        verbose=-1,
        quiet=True
    )

    # Basic shape checks: empty droplets are dropped by default
    n_cells = int((~small_adata.obs["is_empty"]).sum())
    assert isinstance(adata_out, ad.AnnData)
    assert adata_out.X.shape == (n_cells, small_adata.n_vars)
    assert not adata_out.obs["is_empty"].any()
    assert (adata_out.obs["z_hat"] >= 1).all()

    # Check outputs were added
    assert "alpha_hat" in adata_out.obs
    assert "z_hat" in adata_out.obs
    assert "p_hat" in adata_out.uns

    # Check file written
    assert out_path.exists()
    assert ad.read_h5ad(out_path).n_obs == n_cells


def test_denoise_keep_empties(small_adata):
    """keep_empties=True keeps every input barcode; False (default) keeps only real cells."""
    kept = denoise(small_adata.copy(), keep_empties=True, max_iter=2, verbose=-1, quiet=True)
    dropped = denoise(small_adata.copy(), max_iter=2, verbose=-1, quiet=True)

    assert kept.n_obs == small_adata.n_obs
    assert list(kept.obs_names) == list(small_adata.obs_names)
    assert kept.obs.loc[kept.obs["is_empty"], "alpha_hat"].eq(1).all()
    assert kept.obs.loc[kept.obs["is_empty"], "z_hat"].eq(-1).all()

    real = ~small_adata.obs["is_empty"].to_numpy()
    assert dropped.n_obs == real.sum()
    assert list(dropped.obs_names) == list(small_adata.obs_names[real])
    assert "raw" in dropped.layers and dropped.layers["raw"].shape == dropped.X.shape
    # the real cells are denoised identically whether or not the empties are kept
    np.testing.assert_allclose(np.asarray(dropped.X), np.asarray(kept.X)[real])

    # inplace=True subsets the passed object itself
    adata = small_adata.copy()
    out = denoise(adata, inplace=True, max_iter=2, verbose=-1, quiet=True)
    assert out is adata
    assert adata.n_obs == real.sum()

def test_denoise_custom_keys(small_adata):
    """Custom obs/var/uns key names give the same result as the defaults and are written back."""
    renamed = small_adata.copy()
    renamed.obs = renamed.obs.rename(columns={"celltype": "ct", "is_empty": "empty"})
    keys = dict(celltype_key="ct", is_empty_key="empty",
                celltype_profile_key="prof", ambient_profile_key="amb", bulk_profile_key="bulk")

    ref = denoise(small_adata.copy(), max_iter=2, verbose=-1, quiet=True)
    out = denoise(renamed, max_iter=2, verbose=-1, quiet=True, **keys)

    np.testing.assert_allclose(np.asarray(out.X.todense() if sp.issparse(out.X) else out.X),
                               np.asarray(ref.X.todense() if sp.issparse(ref.X) else ref.X))
    assert "amb" in out.var and "ambient_profile" not in out.var
    assert "bulk" in out.var and "bulk_profile" not in out.var
    assert {"prof", "celltype_names", "prof_genes"} <= set(out.uns)
    assert "celltype_profile" not in out.uns


def test_denoise_user_bulk_profile(small_adata):
    """A user-provided bulk profile is normalized and used as-is."""
    adata = small_adata.copy()
    adata.var["bulk_profile"] = [2.0, 1.0, 1.0]
    out = denoise(adata, max_iter=2, verbose=-1, quiet=True)
    np.testing.assert_allclose(out.var["bulk_hat"], [0.5, 0.25, 0.25])

    adata = small_adata.copy()
    adata.var["bulk_profile"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        denoise(adata, max_iter=2, verbose=-1, quiet=True)


# -----------------------
# Edge cases
# -----------------------
def test_no_celltype_column_raises(small_adata):
    del small_adata.obs["celltype"]
    with pytest.raises(KeyError):
        infer_celltype_profile(small_adata)


def test_cli_denoise_runs(tmp_path, small_adata, monkeypatch):
    """Smoke test: ensure CLI command dispatches without error."""
    from cellsweep.main import main

    in_path = tmp_path / "input.h5ad"
    out_path = tmp_path / "denoised_cli.h5ad"
    small_adata.write_h5ad(in_path)

    calls = {}

    def fake_denoise(**kwargs):
        calls["kwargs"] = kwargs
        small_adata.write_h5ad(kwargs["adata_out"])
        return small_adata

    monkeypatch.setattr("cellsweep.main.denoise", fake_denoise)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cellsweep",
            "denoise",
            str(in_path),
            "-o",
            str(out_path),
            "--quiet",
            "--max-iter",
            "2",
        ],
    )

    main()

    assert "kwargs" in calls
    assert calls["kwargs"]["adata"] == str(in_path)
    assert calls["kwargs"]["adata_out"] == str(out_path)
    assert out_path.exists()


@pytest.mark.parametrize("command", ["denoise", "denoise_count_matrix"])
def test_cli_denoise_and_legacy_alias(monkeypatch, command):
    """`cellsweep denoise` and legacy `cellsweep denoise_count_matrix` dispatch identically."""
    from cellsweep.main import main

    calls = {}
    monkeypatch.setattr("cellsweep.main.denoise", lambda **kwargs: calls.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["cellsweep", command, "in.h5ad", "-o", "out.h5ad"])

    main()

    assert calls["adata"] == "in.h5ad"
    assert calls["adata_out"] == "out.h5ad"
    assert calls["keep_empties"] is False


def test_cli_keep_empties_flag(monkeypatch):
    from cellsweep.main import main

    calls = {}
    monkeypatch.setattr("cellsweep.main.denoise", lambda **kwargs: calls.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["cellsweep", "denoise", "in.h5ad", "-o", "out.h5ad", "--keep-empties"])

    main()

    assert calls["keep_empties"] is True


def test_python_denoise_alias():
    import cellsweep

    assert cellsweep.denoise is cellsweep.denoise_count_matrix
