# io.py
# Writen by Thomas Hilder

"""
Utility functions for io and CLI.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import scipy.constants as const
from astropy.io import fits
from numpy.typing import NDArray

from .settings import Settings


# Function to generate parser for CLI
def get_parser() -> argparse.ArgumentParser:
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
        help="fit either a single Gaussian (1, default), summed two component Gaussian (2) or max two component Gaussian (3) profile to each pixel",
        metavar="n_model",
        choices=[1, 2, 3],
        default=Settings.DEFAULT_MODEL,
    )
    # Optional argument to downsample cube
    parser.add_argument(
        "-d",
        "--downsample",
        type=int,
        help="downsample cube using block means, where the block size is n_pix by n_pix",
        metavar="n_pix",
        choices=range(2, 32 + 1),
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


# Function to check whether previous results with the same settings exist in cwd
def previous_results_exist(settings: Settings) -> None:
    if settings.overwrite:
        pass
    else:
        if len(glob.glob(f"*{settings.output_fname_base}*")) > 0:
            raise Exception(
                "Results with the chosen settings already exist in the current working directory. Run with --overwrite in CLI or overwrite=True to run anyway and overwrite them."
            )
        else:
            pass


# Function to read in data from cube
def read_cube(file: Path) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    # Open file with astropy
    with fits.open(name=file) as hdu_list:
        # Get image
        image = hdu_list[0].data
        # Get get axes in sky coordinates
        x_axis, y_axis = sky_coordinates(hdu_list[0].header)
        # Get velocity axis
        v_axis = velocity_axis(hdu_list[0].header)
    # Return image and axes
    return v_axis, x_axis, y_axis, image


# Function to get sky coordinates from fits header
def sky_coordinates(header: fits.header) -> tuple[NDArray, NDArray]:
    # Number of pixels in each dimension
    nx, ny = header["NAXIS1"], header["NAXIS2"]
    # Centre pixels in each dimension
    cx, cy = header["CRPIX1"], header["CRPIX2"]
    # Get pixelscale in arcseconds per pixel
    pix_scale = header["CDELT2"] * 3600
    # Assemble coordinate arrays
    x_axis = (np.arange(1, nx + 1) - cx) * pix_scale * -1
    y_axis = (np.arange(1, ny + 1) - cy) * pix_scale
    # Return coordinates
    return x_axis, y_axis


# Function to get velocity axis from fits header
def velocity_axis(header: fits.header) -> NDArray:
    """Adapted from bettermoments by Richard Teague: github.com/richteague/bettermoments"""

    def read_spectral_axis(header: fits.header) -> NDArray:
        # Number of pixels in spectral dimension
        ns = header["NAXIS3"]
        # Centre pixel in spectral dimension
        cs = header["CRPIX3"]
        # Assemble coordinate array in pixels
        s_ax_pix = np.arange(1, ns + 1, dtype=float) - cs
        # Convert to either Hz or m/s, with explicit type annotation for mypy
        s_ax: NDArray = header["CRVAL3"] + s_ax_pix * header["CDELT3"]
        return s_ax

    def read_rest_frequency(header: fits.header) -> float:
        # Have to try multiple header names
        try:
            nu = header["RESTFREQ"]
        except KeyError:
            try:
                nu = header["RESTFRQ"]
            except KeyError:
                nu = header["CRVAL3"]
        return float(nu)

    # If unit of spectral axis is frequency
    if "freq" in header["CTYPE3"].lower():
        # Read frequencies and rest frequency
        spec_ax = read_spectral_axis(header)
        nu = read_rest_frequency(header)
        # Convert to velocities, with explicit type annotation for mypy
        v_ax: NDArray = (nu - spec_ax) * const.c / nu
        return v_ax
    # If unit is a velocity
    else:
        return read_spectral_axis(header)


def write_moments(
    moments: tuple[NDArray, NDArray, NDArray, NDArray], settings: Settings
) -> None:
    # TODO: Write individuals moments as fits files instead of whole results
    # as one .npz format
    v, x, y, stats = moments
    np.savez(f"summary_{settings.output_fname_base}", v=v, x=x, y=y, stats=stats)
