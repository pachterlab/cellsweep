"""Denoising count matrices using a Poisson + Negative Binomial model."""

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
import anndata as ad
from .utils import setup_logger

def denoise_counts_poisson_nb(C, empty_threshold: int = 10, eps: float = 0.5, tol: float = 1e-4, max_iter: int = 100, C_out: str = None, verbose = 0, quiet = False) -> pd.DataFrame:
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

    # Identify empty vs non-empty droplets
    total_counts = C.sum(axis=1)
    empty_idx = total_counts < empty_threshold
    cell_idx = ~empty_idx

    # 1. Initial ambient Poisson λ_g
    lambda_g = C.loc[empty_idx].mean(axis=0) + eps

    # 2. Initial NB estimates for non-empty droplets
    X_cell = C.loc[cell_idx] - lambda_g
    X_cell[X_cell < 0] = 0
    mu_g = X_cell.mean(axis=0)
    var_g = X_cell.var(axis=0)
    r_g = np.maximum((mu_g ** 2) / (var_g - mu_g + eps), eps)  # avoid negatives

    # 3. EM refinement
    C_cell = C.loc[cell_idx].copy()
    prev_mu, prev_r, prev_lambda = None, None, None

    for i in range(max_iter):
        # ---------- E-step ----------
        # Expected ambient (Poisson) counts
        noise_exp = poisson.mean(lambda_g)
        signal_exp = C_cell - noise_exp
        signal_exp[signal_exp < 0] = 0

        # ---------- M-step ----------
        # Update NB parameters (signal component)
        mu_g_new = signal_exp.mean(axis=0)
        var_g_new = signal_exp.var(axis=0)
        r_g_new = np.maximum((mu_g_new ** 2) / (var_g_new - mu_g_new + eps), eps)

        # Update Poisson λ_g from low-count quantile (adaptive)
        lambda_update = np.clip(signal_exp.quantile(0.05, axis=0), 0, None)
        lambda_g_new = 0.7 * lambda_g + 0.3 * lambda_update  # exponential moving average

        # ---------- Convergence check ----------
        if prev_mu is not None:
            mu_diff = np.nanmean(np.abs(mu_g_new - prev_mu) / (prev_mu + eps))
            r_diff = np.nanmean(np.abs(r_g_new - prev_r) / (prev_r + eps))
            lam_diff = np.nanmean(np.abs(lambda_g_new - prev_lambda) / (prev_lambda + eps))
            delta = max(mu_diff, r_diff, lam_diff)
            if delta < tol:
                if verbose:
                    logger.info(f"Converged at iteration {i+1} (Δ={delta:.2e})")
                break

        # ---------- Update ----------
        mu_g, r_g, lambda_g = mu_g_new, r_g_new, lambda_g_new
        prev_mu, prev_r, prev_lambda = mu_g.copy(), r_g.copy(), lambda_g.copy()

        if verbose and i > 0 and (i+1) % 10 == 0:
            logger.info(f"Iteration {i+1:02d}: Δμ={mu_diff:.2e}, Δr={r_diff:.2e}, Δλ={lam_diff:.2e}")

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
