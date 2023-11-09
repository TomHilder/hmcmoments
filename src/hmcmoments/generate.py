# generate.py
# Written by Thomas Hilder

"""
TODO: Add description
"""

import matplotlib.pyplot as plt
from numpy.typing import NDArray

from .format import format_moments
from .image import downsample_cube, trim_cube
from .io import previous_results_exist, read_cube
from .mcmc import do_mcmc_image
from .settings import Settings


def generate_moments(settings: Settings) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    # Check if previous results exist
    previous_results_exist(settings)
    # Read data
    v_axis, x_axis, y_axis, image = read_cube(settings.file)
    # Trim image to region where the disc is
    x_axis, y_axis, image = trim_cube(x_axis, y_axis, image)
    # Downsample the image using
    if settings.downsample > 1:
        x_axis, y_axis, image = downsample_cube(
            x_axis, y_axis, image, block_size=settings.downsample
        )
    # Preview data for user
    # plt.title("Preview of data for inference (peak intensity)")
    # plt.imshow(np.nanmax(image, axis=0), origin="lower")
    # plt.show()
    # Do MCMC on all line profiles in downsampled image
    summary_statistics = do_mcmc_image(image, v_axis, settings)
    # Format results into moment maps and uncertainties
    return v_axis, x_axis, y_axis, format_moments(summary_statistics)
