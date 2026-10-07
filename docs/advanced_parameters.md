# Advanced EM parameters

{func}`cellsweep.denoise` accepts extra keyword arguments (`**em_kwargs`) that tune the EM
algorithm. Most users will not need to change them; they are exposed for tuning stability
and convergence on unusual datasets. On the command line, pass them as flags of the same
name with underscores replaced by dashes (e.g. `--init-alpha 0.6`).

These values are not range-checked, so stay within the documented bounds. An unknown
keyword raises a `TypeError`.

```python
cellsweep.denoise(adata, init_alpha=0.6, max_iter=3000)
```

## Initialization

`init_alpha` : float, default 0.7, in [0.1, 0.9]
: Starting ambient fraction for each cell when `adata.obs["init_alpha"]` is absent. A high
  value lets the ambient component claim ambient-explainable counts first, so cell-type
  profiles are built mostly from counts the ambient profile cannot explain. The likelihood
  is nearly flat along directions that trade cell-type expression against ambient
  contamination, so this starting value picks among near-equal solutions. Low values can
  leave ambient-shaped expression in the cell-type profiles. Values close to `alpha_cap`
  can strip a cell type's own broadly expressed genes from its profile when that cell type
  is the main source of the ambient RNA (e.g. proximal tubule in kidney), overestimating
  its contamination. Keep it well above the expected contamination rate.

`init_beta` : float, default 0.01
: Starting global (bulk) contamination fraction. Bulk contamination is generally on the
  order of 1%.

`celltype_profile_key` : str, default `"celltype_profile"`
: Key in `adata.uns` holding the initial cell-type profiles (K × G), with the matching
  labels in `adata.uns["celltype_names"]`. If either is missing, both are inferred from
  the cell-type means and written there.

`ambient_profile_key` : str, default `"ambient_profile"`
: Column in `adata.var` holding the initial ambient profile. If missing, it is estimated
  (from the empty droplets when `freeze_ambient_profile=True`, otherwise from the cell-type
  profiles) and written there.

`bulk_profile_key` : str, default `"bulk_profile"`
: Column in `adata.var` holding the bulk (global) contamination profile: one distribution
  over genes, normalized to sum to 1 and held fixed during training. If missing, it is
  estimated from the summed counts of all barcodes (smoothed by `bulk_lambda`) and written
  there.

## Ambient–cell-type separation

`alpha_cap` : float, default 0.9, in [0, 1]
: During burn-in, a cell's ambient fraction may not exceed this value. Barcodes that try to
  are excluded from updating the cell-type profiles and may change cell type. Disabled when
  `freeze_ambient_profile=False`.

`repulsion_strength` : float, default 1e-3, ≥ 0 (values above 1e-3 are untested)
: Strength of repulsion between the ambient and cell-type profiles in the M-step. Higher
  values separate them more. With the default `max_frac_gene_repulsion`, results are
  stable from 2e-5 to 1e-3. Disabled when `freeze_ambient_profile=False`.

`max_frac_gene_repulsion` : float, default 0.25, in (0, 1]
: Largest fraction of each cell-type profile entry that repulsion can remove in one
  iteration. Together with `repulsion_strength` this sets the effective repulsion. Values
  well below 0.25 let cell-type profiles re-absorb ambient-shaped counts over long runs.
  Values of 0.3 and above (with `repulsion_strength` ≥ ~7e-5) can over-correct small cell
  types that express genes abundant in the ambient profile (e.g. DCs in PBMCs). Disabled
  when `freeze_ambient_profile=False`.

## Global contamination prior

`beta_prior_mode` : float, default 0.01, in [0, 1]
: Mode of the Beta prior on the global contamination fraction β. The likelihood only weakly
  identifies β, so the prior keeps its estimate from drifting with the number of iterations.

`beta_prior_strength` : float, default 1e-2, ≥ 0
: Weight of the β prior as a fraction of total counts, so its influence does not depend on
  dataset size. 0 disables the prior (maximum-likelihood β); large values fix β at
  `beta_prior_mode`.

## Smoothing pseudocounts

Each pseudocount is divided by the number of genes G. Higher values give smoother profiles.

`celltype_lambda` : float, default 50, ≥ 0
: Pseudocount for cell-type profile updates.

`ambient_lambda` : float, default 50, ≥ 0
: Pseudocount for the initial and iterative ambient profile updates.

`bulk_lambda` : float, default 10, ≥ 0
: Pseudocount for the bulk profile update.

## Convergence

`max_iter` : int, default 2000, > 1
: Maximum number of EM iterations.

`del0_ll_tol` : float, default 1e-3, > 0
: Change in log-likelihood, relative to the first step, below which the log-likelihood is
  considered stable. Burn-in (the alpha cap and cell-type reassignment) ends once the
  log-likelihood and the cell-type assignments are both stable.

`min_ll_tol` : float, default 1e-6, > 0
: Change in log-likelihood, relative to the current step, below which it is considered
  stable. Caps `del0_ll_tol` at the edge of floating-point precision.

`burnin_patience` : int, default 10, ≥ 0
: Burn-in also requires that no cell has changed cell type for this many consecutive
  iterations. On heavily contaminated data the log-likelihood can stabilize while poorly fit
  cells are still being reassigned. 0 ends burn-in on log-likelihood stability alone.

`burnin_max_iter` : int, default 500, > 0
: Maximum number of burn-in iterations.

`tol_p` : float, default 1e-4, > 0
: Training stops when the largest change in the cell-type profiles falls below this value
  (together with `tol_f`).

`tol_f` : float, default 1e-4, > 0
: Training stops when the largest change in f = (1 − β)·α + β falls below this value
  (together with `tol_p`).

## Numerical stability

`eps` : float, default 1e-12, > 0
: Guards against division by zero.

`log_eps` : float, default 1e-300, > 0
: Guards against log(0).
