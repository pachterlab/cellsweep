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
    Function containing argparse parsers and arguments to allow the use of cellmender from the terminal (as cellmender).
    """

    parent_parser = argparse.ArgumentParser(description=f"cellmender v{__version__}", add_help=False)  # Define parent parser
    parent_subparsers = parent_parser.add_subparsers(dest="command")  # Initiate subparsers
    parent = argparse.ArgumentParser(add_help=False)

    # Add custom help argument to parent parser
    parent_parser.add_argument("-h", "--help", action="store_true", help="Print manual.")
    # Add custom version argument to parent parser
    parent_parser.add_argument("-v", "--version", action="store_true", help="Print version.")

    denoise_count_matrix_desc = "Denoise count matrix using CellMender."

    parser_denoise_count_matrix = parent_subparsers.add_parser(
        "denoise_count_matrix",
        parents=[parent],
        description=denoise_count_matrix_desc,
        help=denoise_count_matrix_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )

    # adata, adata_out="adata_straightened.h5ad", max_iter=40, beta=0.03, eps=1e-9, empty_droplet_method="threshold", umi_cutoff=None, expected_cells=None, cell_ambient_fraction=0.01, empty_droplet_celltype_name="Empty Droplet", round_counts=True, verbose=0, quiet=False, log_file=None
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
        default=40,
        help="Maximum number of training iterations.",
    )
    parser_denoise_count_matrix.add_argument(
        "--beta",
        type=float,
        default=0.03,
        help="Weight of KL divergence term in loss function.",
    )
    parser_denoise_count_matrix.add_argument(
        "--eps",
        type=float,
        default=1e-9,
        help="Small value to avoid division by zero errors.",
    )
    parser_denoise_count_matrix.add_argument(
        "--empty_droplet_method",
        type=str,
        default="threshold",
        choices=["threshold", "expected_cells"],
        help="Method to identify empty droplets. 'threshold' uses umi_cutoff, 'expected_cells' uses expected_cells.",
    )
    parser_denoise_count_matrix.add_argument(
        "--umi_cutoff",
        type=int,
        default=None,
        help="UMI count threshold to identify empty droplets when using 'threshold' method. Required if 'is_empty' column is not present in adata.obs and 'expected_cells' not provided.",
    )
    parser_denoise_count_matrix.add_argument(
        "--expected_cells",
        type=int,
        default=None,
        help="Expected number of cells to identify empty droplets when using 'expected_cells' method. Required if 'is_empty' column is not present in adata.obs and 'umi_cutoff' not provided.",
    )
    parser_denoise_count_matrix.add_argument(
        "--cell_ambient_fraction",
        type=float,
        default=0.01,
        help="Estimated ambient RNA fraction in cell-containing droplets.",
    )
    parser_denoise_count_matrix.add_argument(
        "--empty_droplet_celltype_name",
        type=str,
        default="Empty Droplet",
        help="Cell type name to assign to empty droplets.",
    )
    parser_denoise_count_matrix.add_argument(
        "--round_counts",
        action="store_true",
        help="Round denoised counts to nearest integer.",
    )
    parser_denoise_count_matrix.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity (default logging.WARNING, -v logging.INFO, -vv for logging.DEBUG)"
    )
    parser_denoise_count_matrix.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file to save output messages.",
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
            beta=args.beta,
            eps=args.eps,
            empty_droplet_method=args.empty_droplet_method,
            umi_cutoff=args.umi_cutoff,
            expected_cells=args.expected_cells,
            cell_ambient_fraction=args.cell_ambient_fraction,
            empty_droplet_celltype_name=args.empty_droplet_celltype_name,
            round_counts=args.round_counts,
            verbose=args.verbose,
            log_file=args.log_file,
        )
        
