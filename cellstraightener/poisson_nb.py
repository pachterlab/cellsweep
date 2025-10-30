"""Denoising count matrices using a Poisson + Negative Binomial model."""

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom

def denoise_counts_poisson_nb(C: pd.DataFrame, empty_threshold: int = 10, eps: float = 0.5):
    """
    Denoise a count matrix by modeling ambient noise as Poisson and cell signal as Negative Binomial.
    Args:
        C: pd.DataFrame (droplets x genes)
        empty_threshold: droplets below this total count are treated as empty
        eps: small positive floor to prevent division by zero
    Returns:
        C_denoised: pd.DataFrame (cells x genes)
    """
    C = C.copy()
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
    for _ in range(5):  # iterate EM a few times
        # E-step: expected noise
        noise_exp = poisson.mean(lambda_g)
        signal_exp = C_cell - noise_exp
        signal_exp[signal_exp < 0] = 0

        # M-step: re-estimate NB parameters
        mu_g = signal_exp.mean(axis=0)
        var_g = signal_exp.var(axis=0)
        r_g = (mu_g ** 2) / (var_g - mu_g + eps)

    # 4 Denoised counts (expected signal)
    C_denoised = (C.loc[cell_idx] - lambda_g).clip(lower=0)
    return C_denoised
