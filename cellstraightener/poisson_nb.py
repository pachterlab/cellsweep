"""Denoising count matrices using a Poisson + Negative Binomial model."""

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
import anndata as ad
from .utils import setup_logger

def denoise_counts_poisson_nb(C, empty_threshold: int = 10, eps: float = 0.5, max_iter: int = 100, C_out: str = None, verbose = 0, quiet = False) -> pd.DataFrame:
    """
    Denoise a count matrix by modeling ambient noise as Poisson and cell signal as Negative Binomial.
    Args:
        C: pd.DataFrame (droplets x genes) OR anndata OR path to anndata
        empty_threshold: droplets below this total count are treated as empty
        eps: small positive floor to prevent division by zero
    Returns:
        C_denoised: pd.DataFrame (cells x genes)
    """
    logger = setup_logger(verbose=verbose, quiet=quiet)

    input_format = None
    if isinstance(C, str):
        input_format = "anndata_file"
        if not C.endswith(".h5ad") and not C.endswith(".h5"):
            raise ValueError("If C is a string, it must be a path to an .h5ad or .h5 file.")
        adata = ad.read_h5ad(C) if C.endswith(".h5ad") else ad.read_h5(C)
        C = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
    elif isinstance(C, ad.AnnData):
        input_format = "anndata"
        C = pd.DataFrame(C.X.toarray(), index=C.obs_names, columns=C.var_names)
    elif isinstance(C, pd.DataFrame):
        input_format = "dataframe"
        C = C.copy()
    else:
        raise ValueError("C must be a pd.DataFrame, anndata, or path to an .h5ad/.h5 file.")

    total_counts = C.sum(axis=1)
    empty_idx = total_counts < empty_threshold
    cell_idx = ~empty_idx

    # 1 Estimate Poisson λ_g from empty droplets
    lambda_g = C.loc[empty_idx].mean(axis=0) + eps

    # 2 Estimate initial NB parameters from non-empty droplets
    X_cell = C.loc[cell_idx] - lambda_g  # subtract ambient estimate
    X_cell[X_cell < 0] = 0
    mu_g = X_cell.mean(axis=0)
    var_g = X_cell.var(axis=0)
    r_g = (mu_g ** 2) / (var_g - mu_g + eps)  # method-of-moments for NB

    # 3 EM refinement
    C_cell = C.loc[cell_idx].copy()
    tol = 1e-4  # relative tolerance for convergence
    prev_mu, prev_r = None, None

    for i in range(max_iter):  # max iterations
        # E-step
        noise_exp = poisson.mean(lambda_g)
        signal_exp = C_cell - noise_exp
        signal_exp[signal_exp < 0] = 0

        # M-step
        mu_g_new = signal_exp.mean(axis=0)
        var_g_new = signal_exp.var(axis=0)
        r_g_new = (mu_g_new ** 2) / (var_g_new - mu_g_new + eps)

        # check convergence (element-wise relative change)
        if prev_mu is not None:
            mu_diff = np.mean(np.abs(mu_g_new - prev_mu) / (prev_mu + eps))
            r_diff = np.mean(np.abs(r_g_new - prev_r) / (prev_r + eps))
            if mu_diff < tol and r_diff < tol:
                if verbose:
                    logger.info(f"EM converged at iteration {i+1} (Δμ={mu_diff:.2e}, Δr={r_diff:.2e})")
                break

        prev_mu, prev_r = mu_g_new.copy(), r_g_new.copy()
        mu_g, r_g = mu_g_new, r_g_new
    else:
        if verbose:
            logger.info("Reached max EM iterations without convergence.")


    # 4 Denoised counts (expected signal)
    C_denoised = (C.loc[cell_idx] - lambda_g).clip(lower=0)
    if input_format == "anndata_file" or input_format == "anndata":
        adata_denoised = ad.AnnData(X=C_denoised.values, obs=C_denoised.index.to_frame(), var=C_denoised.columns.to_frame())
        if C_out:
            adata_denoised.write_h5ad(C_out)
        return adata_denoised
    elif input_format == "dataframe":
        if C_out:
            C_denoised.to_csv(C_out)
        return C_denoised
    else:
        raise ValueError("Unexpected input format.")
