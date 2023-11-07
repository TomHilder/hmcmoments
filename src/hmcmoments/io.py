# io.py
# Writen by Thomas Hilder

"""
Utility functions for io and CLI.
"""

import argparse

from .settings import Settings


def get_parser():
    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description="Generate moment maps and corresponding posteriors by performing inference on line profiles with Hamiltonian Monte Carlo.",
    )
    # Positional argument for cube.fits file
    parser.add_argument(
        "filename",
        type=str,
        help=".fits file containing cube",
    )
    # Optional argument for model to fit (default is 1 Gaussian component)
    parser.add_argument(
        "-m",
        "--model",
        type=int,
        help="fit either a single (1, default) or double component (2) Gaussian profile to each pixel",
        metavar="n_model",
        choices=[1, 2],
        default=Settings.DEFAULT_MODEL,
    )
    # Optional argument to downsample cube
    parser.add_argument(
        "-d",
        "--downsample",
        type=int,
        help="downsample cube using block means, where the block size is n_pix by n_pix",
        metavar="n_pix",
        choices=range(2, 11),
        default=Settings.DEFAULT_DOWNSAMPLE,
    )
    # Optional argument to choose the number of cores
    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        help="number of cores used by program (default is all physical cores, which may not be available to you on a cluser)",
        metavar="n_cores",
        default=Settings.DEFAULT_CORES,
    )
    # Optional argument to overwrite existing data in cwd
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="overwrite existing output data in current working directory",
        default=Settings.DEFAULT_OVERWRITE,
    )
    # Return parser
    return parser
