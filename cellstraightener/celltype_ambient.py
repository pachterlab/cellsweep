"""Denoising count matrices using a Poisson + Negative Binomial model."""

import numpy as np
import pandas as pd
import logging
from scipy.stats import poisson, nbinom
from .utils import setup_logger

def denoise_counts_celltype_ambient():
    pass

# def denoise_counts_celltype_ambient(sim, K=3, max_iter=40, beta=0.3, verbose=0, quiet=False):
#     """
#     EM on *real* cells only, with:
#       - ambient fixed to the true ambient
#       - empty-vs-real fixed to the true empties
#     This is to test whether the p_k and alpha_i parts are behaving.
#     """
    
#     logger = setup_logger(verbose=verbose, quiet=quiet)

#     X = sim["X"].astype(float)
#     N, G = X.shape
#     eps = 1e-9

#     a = sim["ambient"].copy()           # FIXED ambient
#     is_empty = sim["is_empty"].copy()   # FIXED empties
#     z_true = sim["z"].copy()

#     # work only on real cells
#     real_mask = ~is_empty
#     Xr = X[real_mask]           # (Nr, G)
#     Nr = Xr.shape[0]

#     # initial cell-type profiles: from truth
#     p = sim["p_celltypes"].copy()   # (K, G)

#     # initial responsibilities: from truth if available
#     gamma_type = np.zeros((Nr, K))
#     for j, i in enumerate(np.where(real_mask)[0]):
#         if z_true[i] >= 0:
#             gamma_type[j, z_true[i]] = 1.0
#         else:
#             gamma_type[j] = 1.0 / K

#     # initial alpha: from truth
#     alpha = sim["alpha"][real_mask].copy()

#     loglike_prev = -np.inf

#     for it in range(max_iter):
#         m = p.mean(axis=0)

#         # --- E step on real cells ---
#         log_p_type = np.zeros((Nr, K))
#         for k in range(K):
#             pi_j = alpha[:, None] * a + (1 - alpha)[:, None] * (
#                 (1 - beta) * p[k] + beta * m
#             )
#             pi_j = np.clip(pi_j, eps, 1.0)
#             pi_j /= pi_j.sum(axis=1, keepdims=True)
#             log_p_type[:, k] = np.sum(Xr * np.log(pi_j), axis=1)

#         # normalize over k
#         log_p_type -= log_p_type.max(axis=1, keepdims=True)
#         r = np.exp(log_p_type)
#         r /= r.sum(axis=1, keepdims=True)
#         gamma_type = r

#         # --- M step on real cells ---
#         # update p_k
#         for k in range(K):
#             p[k] = (gamma_type[:, k][:, None] * Xr).sum(axis=0) + 1.0  # smoothing
#             p[k] /= p[k].sum()

#         # update alpha_j by 1D search
#         for j in range(Nr):
#             mix_cell = (1 - beta) * (gamma_type[j] @ p) + beta * p.mean(axis=0)
#             best_alpha, best_ll = alpha[j], -np.inf
#             for a_try in np.linspace(0, 1, 31):
#                 pi_try = a_try * a + (1 - a_try) * mix_cell
#                 pi_try = np.clip(pi_try, eps, 1.0)
#                 ll = np.sum(Xr[j] * np.log(pi_try))
#                 if ll > best_ll:
#                     best_ll, best_alpha = ll, a_try
#             alpha[j] = best_alpha

#         # log-likelihood on real cells
#         loglike = np.sum(np.log(np.exp(log_p_type).sum(axis=1) + eps))
#         if verbose:
#             print(f"Iter {it+1:2d}: logL(real)={loglike:.3f}")
#         if np.abs(loglike - loglike_prev) < 1e-4:
#             break
#         loglike_prev = loglike

#     # build outputs back to full size
#     alpha_hat = np.zeros(N)
#     alpha_hat[real_mask] = alpha
#     alpha_hat[~real_mask] = 1.0  # empties

#     # predicted labels for real cells
#     z_hat = np.full(N, -1)
#     z_hat[real_mask] = np.argmax(gamma_type, axis=1)

#     return {
#         "ambient_hat": a,
#         "p_hat": p,
#         "alpha_hat": alpha_hat,
#         "z_hat": z_hat,
#         "is_empty_hat": is_empty,  # fixed
#         "loglike": loglike,
#     }


# # ---- run it ----
# fit = denoise_counts_celltype_ambient(sim, K=3, max_iter=40, beta=0.3, verbose=True)

# print("\n=== Ambient (should match true exactly) ===")
# print("est:", np.round(fit["ambient_hat"], 3))
# print("tru:", np.round(sim["ambient"], 3))

# print("\n=== Cell-type distributions ===")
# for k in range(3):
#     print(f"\nType {k}")
#     print("est:", np.round(fit["p_hat"][k], 3))
#     print("tru:", np.round(sim["p_celltypes"][k], 3))

# print("\n=== Alpha (first 10) ===")
# for i in range(10):
#     print(f"cell {i:2d}: est={fit['alpha_hat'][i]:.3f}, tru={sim['alpha'][i]:.3f}, empty={sim['is_empty'][i]}")

# print("\n=== Empty (should match true) ===")
# print("pred empty:", fit["is_empty_hat"].sum(), "true empty:", sim["is_empty"].sum())

# print("\n=== Cell-type assignments (first 20) ===")
# print("pred:", fit["z_hat"][:20])
# print("true:", sim["z"][:20])
