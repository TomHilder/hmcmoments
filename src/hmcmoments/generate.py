# generate.py
# Written by Thomas Hilder

"""
TODO: Add description
"""

import matplotlib.pyplot as plt
import numpy as np

from .image import downsample_cube, estimate_rms, trim_cube
from .io import previous_results_exist, read_cube
from .mcmc import do_mcmc
from .settings import Settings


def generate(settings: Settings) -> None:
    # Check if previous results exist
    previous_results_exist(settings)
    # Read data
    v_axis, x_axis, y_axis, image = read_cube(settings.file)
    # Trim image to region where the disc is
    x_axis, y_axis, image = trim_cube(x_axis, y_axis, image)
    plt.imshow(np.nanmax(image, axis=0), origin="lower")
    plt.show()
    # Downsample image
    if settings.downsample > 1:
        x_axis, y_axis, image = downsample_cube(
            x_axis, y_axis, image, block_size=settings.downsample
        )
    # Do MCMC
    results = do_mcmc(image, v_axis, settings)
