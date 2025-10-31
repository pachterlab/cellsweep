"""Denoising count matrices using a Poisson + Negative Binomial model."""

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
import anndata as ad
from .utils import setup_logger

def denoise_counts_poisson_nb(
    C,
    empty_threshold: int = 10,
    eps: float = 0.5,
    tol: float = 1e-4,
    max_iter: int = 100,
    C_out: str = None,
    verbose: int = 0,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Denoise a count matrix by modeling ambient noise as Poisson and cell signal as Negative Binomial,
    estimated jointly with an EM algorithm.

    Args:
        C: pd.DataFrame (droplets x genes) OR anndata OR path to anndata
        empty_threshold: droplets below this total count are treated as empty
        eps: small positive floor to prevent division by zero
        tol: relative tolerance for convergence
        max_iter: maximum EM iterations
        C_out: optional output file (h5ad or csv)
        verbose: verbosity level (0 = none, 1 = progress)
        quiet: suppress all logs

    Returns:
        Denoised AnnData or DataFrame (same type as input)
    """
    logger = setup_logger(verbose=verbose, quiet=quiet)

    # --------------------------
    # 1. Load / standardize input
    # --------------------------
    input_format = None
    if isinstance(C, str):
        input_format = "anndata_file"
        if not C.endswith((".h5ad", ".h5")):
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

    # --------------------------
    # 2. Identify empty droplets
    # --------------------------
    total_counts = C.sum(axis=1)
    empty_idx = total_counts < empty_threshold
    cell_idx = ~empty_idx

    # --------------------------
    # 3. Initialize parameters
    # --------------------------
    lambda_g = C.loc[empty_idx].mean(axis=0) + eps  # Poisson mean
    X_cell = (C.loc[cell_idx] - lambda_g).clip(lower=0)
    mu_g = X_cell.mean(axis=0) + eps
    var_g = X_cell.var(axis=0) + eps
    r_g = np.maximum(mu_g**2 / (var_g - mu_g + eps), eps)  # NB dispersion
    C_cell = C.loc[cell_idx].copy()

    prev_mu, prev_r, prev_lambda = None, None, None

    # --------------------------
    # 4. EM iterations
    # --------------------------
    for i in range(max_iter):
        # --- E-step ---
        # Compute responsibilities γ_ng = posterior prob(count came from Poisson)
        pois_ll = poisson.pmf(C_cell, lambda_g.values)
        nb_ll = nbinom.pmf(C_cell, n=r_g.values, p=r_g.values / (r_g.values + mu_g.values))
        gamma = pois_ll / (pois_ll + nb_ll + eps)  # shape (cells, genes)

        # Expected noise and signal counts
        C_noise_exp = gamma * C_cell
        C_signal_exp = (1 - gamma) * C_cell

        # --- M-step ---
        lambda_g_new = C_noise_exp.mean(axis=0) + eps
        mu_g_new = C_signal_exp.mean(axis=0) + eps
        var_g_new = C_signal_exp.var(axis=0) + eps
        r_g_new = np.maximum(mu_g_new**2 / (var_g_new - mu_g_new + eps), eps)

        # --- Convergence check ---
        if prev_mu is not None:
            mu_diff = np.nanmean(np.abs(mu_g_new - prev_mu) / (prev_mu + eps))
            r_diff = np.nanmean(np.abs(r_g_new - prev_r) / (prev_r + eps))
            lam_diff = np.nanmean(np.abs(lambda_g_new - prev_lambda) / (prev_lambda + eps))
            delta = max(mu_diff, r_diff, lam_diff)
            if delta < tol:
                if verbose:
                    logger.info(f"Converged at iteration {i+1} (Δ={delta:.2e})")
                break

        mu_g, r_g, lambda_g = mu_g_new, r_g_new, lambda_g_new
        prev_mu, prev_r, prev_lambda = mu_g.copy(), r_g.copy(), lambda_g.copy()

        if verbose and i > 0 and (i+1) % 10 == 0:
            logger.info(f"Iteration {i+1:02d}: Δμ={mu_diff:.2e}, Δr={r_diff:.2e}, Δλ={lam_diff:.2e}")

    else:
        if verbose:
            logger.info("Reached max EM iterations without convergence.")

    # --------------------------
    # 5. Construct denoised output
    # --------------------------
    # Expected signal counts (denoised)
    C_denoised = pd.DataFrame(C_signal_exp, index=C_cell.index, columns=C_cell.columns)

    if input_format in {"anndata_file", "anndata"}:
        adata_denoised = ad.AnnData(
            X=C_denoised.values,
            obs=C_denoised.index.to_frame(),
            var=C_denoised.columns.to_frame(),
        )
        if C_out:
            adata_denoised.write_h5ad(C_out)
        return adata_denoised

    if input_format == "dataframe":
        if C_out:
            C_denoised.to_csv(C_out)
        return C_denoised

    raise ValueError("Unexpected input format.")