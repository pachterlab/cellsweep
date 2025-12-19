"""main function for argparse."""

import argparse
import sys
from .__init__ import __version__
from .celltype_ambient import denoise_count_matrix

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


def main():  # noqa: C901
    """
    Function containing argparse parsers and arguments to allow the use of cellsweep from the terminal (as cellsweep).
    """

    parent_parser = argparse.ArgumentParser(description=f"cellsweep v{__version__}", add_help=False)  # Define parent parser
    parent_subparsers = parent_parser.add_subparsers(dest="command")  # Initiate subparsers
    parent = argparse.ArgumentParser(add_help=False)

    # Add custom help argument to parent parser
    parent_parser.add_argument("-h", "--help", action="store_true", help="Print manual.")
    # Add custom version argument to parent parser
    parent_parser.add_argument("-v", "--version", action="store_true", help="Print version.")

    denoise_count_matrix_desc = "Denoise count matrix using cellsweep."

    parser_denoise_count_matrix = parent_subparsers.add_parser(
        "denoise_count_matrix",
        parents=[parent],
        description=denoise_count_matrix_desc,
        help=denoise_count_matrix_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )

    parser_denoise_count_matrix.add_argument(
        "adata",
        type=str,
        help="Path to input AnnData file (.h5ad) containing raw count matrix in .X.",
    )
    parser_denoise_count_matrix.add_argument(
        "--adata_out",
        type=str,
        default="adata_denoised.h5ad",
        help="Path to output AnnData file (.h5ad) to save denoised count matrix. (default: adata_denoised.h5ad)",
    )
    parser_denoise_count_matrix.add_argument(
        "--max_iter",
        type=int,
        default=500,
        help="Maximum number of EM iterations.",
    )
    parser_denoise_count_matrix.add_argument(
        "--init_alpha",
        type=float,
        default=0.9,
        help="Initial value of alpha_n for each cell. Works better when close to 1.",
    )
    parser_denoise_count_matrix.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Initial beta (percent bulk contamination) value for each cell. Works better when set to a higher number than expected (expected is around 0.05).",
    )
    parser_denoise_count_matrix.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Numerical stability constant to prevent division by zero or log(0).",
    )
    parser_denoise_count_matrix.add_argument(
        "--log_eps",
        type=float,
        default=1e-300,
        help="Numerical stability constant to prevent division by zero or log(0). Lower than eps for log values.",
    )
    parser_denoise_count_matrix.add_argument(
        "--dirichlet_lambda",
        type=float,
        default=10,
        help="Pseudocount. Sometimes used for clipping.",
    )
    parser_denoise_count_matrix.add_argument(
        "--integer_out",
        action="store_true",
        help="If True, rounds denoised counts to nearest integer before saving.",
    )
    parser_denoise_count_matrix.add_argument(
        "-t", "--threads",
        type=int,
        default=1,
        help="number of numba threads",
    )
    parser_denoise_count_matrix.add_argument(
        "--disable_fixed_celltype",
        action="store_false",
        help="Use fixed cell type annotations from adata.obs['cell_type'] if available.",
    )
    parser_denoise_count_matrix.add_argument(
        "--disable_freeze_empty",
        action="store_false",
        help="If True, does not attempt to reestimate empty droplets."
    )
    parser_denoise_count_matrix.add_argument(
        "--disable_freeze_ambient_profile",
        action="store_false",
        help="If True, does not update the ambient profile (a) based on alpha."
    )
    parser_denoise_count_matrix.add_argument(
        "--empty_droplet_method",
        type=str,
        default="threshold",
        help=(
            "Strategy to infer empty droplets if `is_empty` is not present. "
            "Options include: 'threshold', 'quantile', or model-based approaches. "
            "(default: 'threshold')"
        ),
    )
    parser_denoise_count_matrix.add_argument(
        "--ambient_threshold",
        type=float,
        default=0.0,
        help=(
            "Strategy to infer empty droplets if `is_empty` is not present. Options may include 'threshold', 'quantile', or model-based approaches."
        ),
    )
    parser_denoise_count_matrix.add_argument(
        "--umi_cutoff",
        type=int,
        default=None,
        help=(
            "Optional absolute UMI count threshold for classifying droplets as empty. "
            "(default: None)"
        ),
    )
    parser_denoise_count_matrix.add_argument(
        "--expected_cells",
        type=int,
        default=None,
        help=(
            "Expected number of real cells, used when estimating thresholds. "
            "(default: None)"
        ),
    )
    parser_denoise_count_matrix.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help=(
            "Relative change in likelihood below which training stops. "
            "(default: 1e-6)"
        ),
    )
    parser_denoise_count_matrix.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed. (default: 42)",
    )
    parser_denoise_count_matrix.add_argument(
        "--verbose",
        type=int,
        default=0,
        help=(
            "Verbosity level: 2 debug, 1 info, 0 warning, -1 error, -2 critical. "
            "(default: 0)"
        ),
    )
    parser_denoise_count_matrix.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppresses most log output when True. (default: False)",
    )
    parser_denoise_count_matrix.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path to save EM iteration logs. (default: None)",
    )

    args, unknown_args = parent_parser.parse_known_args()

    # Help return
    if args.help:
        # Retrieve all subparsers from the parent parser
        subparsers_actions = [action for action in parent_parser._actions if isinstance(action, argparse._SubParsersAction)]
        for subparsers_action in subparsers_actions:
            # Get all subparsers and print help
            for choice, subparser in subparsers_action.choices.items():
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
        "denoise_count_matrix": parser_denoise_count_matrix,
    }
    
    if len(sys.argv) == 2:
        if sys.argv[1] in command_to_parser:
            command_to_parser[sys.argv[1]].print_help(sys.stderr)
        else:
            parent_parser.print_help(sys.stderr)
        sys.exit(1)
    
    if args.command == "denoise_count_matrix":
        denoise_count_matrix(
            adata=args.adata,
            adata_out=args.adata_out,
            max_iter=args.max_iter,
            init_alpha=args.init_alpha,
            beta=args.beta,
            eps=args.eps,
            log_eps=args.log_eps,
            dirichlet_lambda=args.dirichlet_lambda,
            integer_out=args.integer_out,
            threads=args.threads,
            fixed_celltype=args.disable_fixed_celltype,
            freeze_empty=args.disable_freeze_empty,
            freeze_ambient_profile=args.disable_freeze_ambient_profile,
            empty_droplet_method=args.empty_droplet_method,
            ambient_threshold=args.ambient_threshold,
            umi_cutoff=args.umi_cutoff,
            expected_cells=args.expected_cells,
            tol=args.tol,
            random_state=args.random_state,
            verbose=args.verbose,
            quiet=args.quiet,
            log_file=args.log_file,
        )
        
