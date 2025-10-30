"""Denoising count matrices using a Poisson + Negative Binomial model."""

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom

def denoise_counts_celltype_ambient(C: pd.DataFrame, empty_threshold: int = 10, eps: float = 0.5):
    return C  # Placeholder for the actual implementation