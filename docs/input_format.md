# Input and output format

## Input AnnData

The input AnnData object (or `.h5ad` file) should have the following structure.

`adata.X`
: Raw count matrix (cells × genes), including empty droplets.

`adata.obs["celltype"]`
: Cell-type label for each cell. Labels on empty droplets are ignored. See
  {func}`cellsweep.utils.data_utils.determine_cell_types` for one way to produce these.

`adata.obs["is_empty"]` *(optional)*
: Boolean column marking non-cellular barcodes (empty droplets). If missing, CellSweep
  infers them with the `empty_droplet_method` argument, using `umi_cutoff` or
  `expected_cells` if given.

`adata.obs["init_alpha"]` *(optional)*
: Initial estimate of each cell's ambient contamination fraction. If missing, the
  `init_alpha` argument is used for every cell (see {doc}`advanced_parameters`).

`adata.var["ambient_profile"]` *(optional)*
: Per-gene ambient RNA fraction. If missing, it is estimated from the empty droplets.

`adata.var["bulk_profile"]` *(optional)*
: Per-gene bulk (global) contamination distribution, held fixed during training. If
  missing, it is estimated from the summed counts of all barcodes.

`adata.uns["celltype_profile"]` *(optional)*
: Mean expression for each cell type (K × G). Inferred from the data if missing.

`adata.uns["celltype_names"]` *(optional)*
: Cell-type label for each row of `celltype_profile`. Inferred together with
  `celltype_profile` if either is missing.

`adata.uns["celltype_profile_genes"]` *(optional)*
: Gene names corresponding to the columns of `celltype_profile`.

### Custom column and key names

Some of the names above can be changed. If your AnnData stores these fields under
different names, pass the matching argument to {func}`cellsweep.denoise` (or the
`--<argument>` flag on the command line). The profile keys are advanced EM parameters
(see {doc}`advanced_parameters`).

| Field | Argument | Default |
| --- | --- | --- |
| `adata.obs` cell-type labels | `celltype_key` | `"celltype"` |
| `adata.obs` empty-droplet flags | `is_empty_key` | `"is_empty"` |
| `adata.uns` cell-type profiles | `celltype_profile_key` | `"celltype_profile"` |
| `adata.var` ambient profile | `ambient_profile_key` | `"ambient_profile"` |
| `adata.var` bulk profile | `bulk_profile_key` | `"bulk_profile"` |

```python
cellsweep.denoise(adata, celltype_key="cell_type", is_empty_key="empty")
```

Fields that CellSweep infers (empty-droplet flags, cell-type profiles, ambient and bulk
profiles) are written back under the same names.

(outputs)=
## Output AnnData

{func}`cellsweep.denoise` returns an AnnData object with the denoised counts in `adata.X`
and these added fields:

| Field | Description |
| --- | --- |
| `adata.layers["raw"]` | Raw count matrix |
| `adata.obs["alpha_hat"]` | Estimated ambient contamination fraction per cell |
| `adata.obs["z_hat"]` | Final cell-type assignment per cell |
| `adata.var["ambient_hat"]` | Estimated ambient profile |
| `adata.var["bulk_hat"]` | Estimated global (bulk) noise profile |
| `adata.uns["p_hat"]` | Estimated cell-type profiles (K × G) |
| `adata.uns["beta_hat"]` | Estimated global contamination fraction |
| `adata.uns["loglike"]` | Final relative log-likelihood (not the complete log-likelihood) |

Denoised counts are fractional unless `round_X=True`, which rounds them stochastically
(seeded by `random_state`).

By default the output contains only the real cells: the empty droplets are used to fit the
ambient profile and then dropped. Pass `keep_empties=True` (CLI: `--keep-empties`) to keep
every input barcode; empty droplets then have `alpha_hat = 1` and `z_hat = -1`.
