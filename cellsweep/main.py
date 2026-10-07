"""main function for argparse."""

import argparse
import sys
from .__init__ import __version__
from .model import denoise

# Custom formatter for help messages that preserved the text formatting and adds the default value to the end of the help message
class CustomHelpFormatter(argparse.RawTextHelpFormatter):
    def _get_help_string(self, action):
        help_str = action.help if action.help else ""
        if (
            "%(default)" not in help_str
            and action.default is not argparse.SUPPRESS
            and action.default is not None
            # default information can be deceptive or confusing for boolean flags.
            # For example, `--quiet` says "Does not print progress information. (default: True)" even though
            # the default action is to NOT be quiet (to the user, the default is False).
            and not isinstance(action, argparse._StoreTrueAction)
            and not isinstance(action, argparse._StoreFalseAction)
        ):
            help_str += " (default: %(default)s)"
        return help_str


def build_parser():
    """
    Build the argparse parser for the cellsweep command line interface.

    Returns
    -------
    tuple
        ``(parent_parser, parser_denoise)``.
    """

    parent_parser = argparse.ArgumentParser(description=f"cellsweep v{__version__}", add_help=False)  # Define parent parser
    parent_subparsers = parent_parser.add_subparsers(dest="command")  # Initiate subparsers
    parent = argparse.ArgumentParser(add_help=False)

    # Add custom help argument to parent parser
    parent_parser.add_argument("-h", "--help", action="store_true", help="Print manual.")
    # Add custom version argument to parent parser
    parent_parser.add_argument("-v", "--version", action="store_true", help="Print version.")

    denoise_desc = "Denoise count matrix using cellsweep."

    parser_denoise = parent_subparsers.add_parser(
        "denoise",
        parents=[parent],
        description=denoise_desc,
        help=denoise_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )

    parser_denoise.add_argument(
        "adata",
        type=str,
        help="Path to input AnnData file (.h5ad) containing raw count matrix in .X.",
    )
    parser_denoise.add_argument(
        "-o",
        "--adata-out",
        type=str,
        default="adata_denoised.h5ad",
        help="Path to output AnnData file (.h5ad) to save denoised count matrix.",
    )
    parser_denoise.add_argument(
        "--round-X",
        action="store_true",
        help="If True, rounds denoised counts to nearest integer before saving.",
    )
    parser_denoise.add_argument(
        "--keep-empties",
        action="store_true",
        help="If set, keeps the empty droplets in the output. By default they are removed after denoising, so the output contains only the real cells.",
    )
    parser_denoise.add_argument(
        "-t", "--threads",
        type=int,
        default=1,
        help="number of numba threads",
    )
    parser_denoise.add_argument(
        "--disable-freeze-ambient-profile",
        action="store_false",
        help="If set, models the ambient profile (a) as a mixture of cell-type profiles instead of anchoring it on empty droplets."
    )
    parser_denoise.add_argument(
        "--empty-droplet-method",
        type=str,
        default="threshold",
        choices=["threshold"],
        help="Strategy to infer empty droplets if `is_empty` is not present."
    )
    parser_denoise.add_argument(
        "--umi-cutoff",
        type=int,
        default=None,
        help="Optional absolute UMI count threshold for classifying droplets as empty."
    )
    parser_denoise.add_argument(
        "--expected-cells",
        type=int,
        default=None,
        help="Expected number of real cells, used when estimating thresholds."
    )
    # Advanced EM hyperparameters: hidden from --help (see denoise's
    # "Other Parameters" docstring section for details), but still settable.
    parser_denoise.add_argument(
        "--init-alpha",
        type=float,
        default=0.7,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--init-beta",
        type=float,
        default=0.01,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--celltype-profile-key",
        type=str,
        default="celltype_profile",
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--ambient-profile-key",
        type=str,
        default="ambient_profile",
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--bulk-profile-key",
        type=str,
        default="bulk_profile",
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--alpha-cap",
        type=float,
        default=0.9,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--repulsion-strength",
        type=float,
        default=1e-3,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--max-frac-gene-repulsion",
        type=float,
        default=0.25,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--beta-prior-mode",
        type=float,
        default=0.01,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--beta-prior-strength",
        type=float,
        default=1e-2,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--celltype-lambda",
        type=float,
        default=50,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--ambient-lambda",
        type=float,
        default=50,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--bulk-lambda",
        type=float,
        default=10,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--log-eps",
        type=float,
        default=1e-300,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--del0-ll-tol",
        type=float,
        default=1e-3,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--min-ll-tol",
        type=float,
        default=1e-6,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--burnin-patience",
        type=int,
        default=10,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--burnin-max-iter",
        type=int,
        default=500,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--tol-p",
        type=float,
        default=1e-4,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--tol-f",
        type=float,
        default=1e-4,
        help=argparse.SUPPRESS,
    )
    parser_denoise.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser_denoise.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Verbosity level. Default logging.WARNING, -v logging.INFO, -vv for logging.DEBUG)"
    )
    parser_denoise.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output (overrides any verbose flag)",
    )
    parser_denoise.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to save EM iteration logs.",
    )
    parser_denoise.add_argument(
        "--celltype-key",
        type=str,
        default="celltype",
        help="adata.obs column holding the input cell-type labels.",
    )
    parser_denoise.add_argument(
        "--is-empty-key",
        type=str,
        default="is_empty",
        help="adata.obs column marking non-cellular barcodes; written here if inferred.",
    )

    # backwards compatibility: accept the old underscore spellings (e.g. --max_iter) without listing them in --help
    for option_string, action in list(parser_denoise._option_string_actions.items()):
        if option_string.startswith("--") and "-" in option_string[2:]:
            parser_denoise._option_string_actions.setdefault("--" + option_string[2:].replace("-", "_"), action)

    # backwards compatibility: accept `cellsweep denoise_count_matrix` without listing it in --help
    parent_subparsers._name_parser_map["denoise_count_matrix"] = parser_denoise
    parent_subparsers.metavar = "{denoise}"

    return parent_parser, parser_denoise


def get_parser():
    """Return the top-level cellsweep parser (used by the Sphinx docs)."""
    return build_parser()[0]


def main():  # noqa: C901
    """
    Function containing argparse parsers and arguments to allow the use of cellsweep from the terminal (as cellsweep).
    """

    parent_parser, parser_denoise = build_parser()
    args, unknown_args = parent_parser.parse_known_args()

    # Help return
    if args.help:
        # Retrieve all subparsers from the parent parser
        subparsers_actions = [action for action in parent_parser._actions if isinstance(action, argparse._SubParsersAction)]
        for subparsers_action in subparsers_actions:
            # Get all subparsers and print help
            seen = set()
            for choice, subparser in subparsers_action.choices.items():
                if id(subparser) in seen:  # skip aliases
                    continue
                seen.add(id(subparser))
                print("Subparser '{}'".format(choice))
                print(subparser.format_help())
        sys.exit(1)

    # Version return
    if args.version:
        print(f"varseek version: {__version__}")
        sys.exit(1)

    # Show help when no arguments are given
    if len(sys.argv) == 1:
        parent_parser.print_help(sys.stderr)
        sys.exit(1)
    
    command_to_parser = {
        "denoise": parser_denoise,
        "denoise_count_matrix": parser_denoise,
    }
    
    if len(sys.argv) == 2:
        if sys.argv[1] in command_to_parser:
            command_to_parser[sys.argv[1]].print_help(sys.stderr)
        else:
            parent_parser.print_help(sys.stderr)
        sys.exit(1)
    
    if args.command in ("denoise", "denoise_count_matrix"):
        denoise(
            adata=args.adata,
            adata_out=args.adata_out,
            round_X=args.round_X,
            keep_empties=args.keep_empties,
            threads=args.threads,
            freeze_ambient_profile=args.disable_freeze_ambient_profile,
            empty_droplet_method=args.empty_droplet_method,
            umi_cutoff=args.umi_cutoff,
            expected_cells=args.expected_cells,
            init_alpha=args.init_alpha,
            init_beta=args.init_beta,
            celltype_profile_key=args.celltype_profile_key,
            ambient_profile_key=args.ambient_profile_key,
            bulk_profile_key=args.bulk_profile_key,
            alpha_cap=args.alpha_cap,
            repulsion_strength=args.repulsion_strength,
            max_frac_gene_repulsion=args.max_frac_gene_repulsion,
            beta_prior_mode=args.beta_prior_mode,
            beta_prior_strength=args.beta_prior_strength,
            celltype_lambda=args.celltype_lambda,
            ambient_lambda=args.ambient_lambda,
            bulk_lambda=args.bulk_lambda,
            eps=args.eps,
            log_eps=args.log_eps,
            max_iter=args.max_iter,
            del0_ll_tol=args.del0_ll_tol,
            min_ll_tol=args.min_ll_tol,
            burnin_patience=args.burnin_patience,
            burnin_max_iter=args.burnin_max_iter,
            tol_p=args.tol_p,
            tol_f=args.tol_f,
            random_state=args.random_state,
            inplace=False,  # No need to copy adata because we are loading it from a file path and saving to a new file path, so there is no risk of modifying the input in-place.
            verbose=args.verbose,
            quiet=args.quiet,
            log_file=args.log_file,
            celltype_key=args.celltype_key,
            is_empty_key=args.is_empty_key,
        )
        
