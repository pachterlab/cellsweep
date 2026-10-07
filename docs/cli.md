# Command line interface

Installing CellSweep adds a `cellsweep` command. Its main subcommand is `denoise`, which
reads an `.h5ad` file and writes the denoised result to another `.h5ad` file.

```bash
cellsweep denoise -o adata_cellsweep.h5ad adata_raw.h5ad
```

The advanced EM hyperparameters (`--init-alpha`, `--max-iter`, ...) are hidden from
`--help` but can still be passed on the command line. They are described in
{doc}`advanced_parameters`.

```{argparse}
:module: cellsweep.main
:func: get_parser
:prog: cellsweep
:nodefault:
```
