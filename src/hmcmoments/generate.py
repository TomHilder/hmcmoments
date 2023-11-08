# generate.py
# Written by Thomas Hilder

"""
TODO: Add description
"""

from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import block_reduce

from .image import downsample_cube, estimate_rms, trim_cube
from .io import previous_results_exist, read_cube
from .settings import Settings


def generate(settings: Settings) -> None:
    # INITIALISATION

    # Check if previous results exist
    previous_results_exist(settings=settings)

    # Read data
    x_axis, y_axis, image = read_cube(file=settings.file)

    plt.contourf(x_axis, y_axis, np.nanmax(image, axis=0), levels=100)
    plt.axis("scaled")
    plt.xlim(x_axis.max(), x_axis.min())
    plt.show()

    # TRIM CUBE

    trimmed_x_axis, trimmed_y_axis, trimmed_image = trim_cube(
        x_axis=x_axis, y_axis=y_axis, image=image
    )

    # DOWNSAMPLE

    if settings.downsample > 1:
        ds_trimmed_x_axis, ds_trimmed_y_axis, ds_trimmed_image = downsample_cube(
            x_axis=trimmed_x_axis,
            y_axis=trimmed_y_axis,
            image=trimmed_image,
            block_size=settings.downsample,
        )

    else:
        ds_trimmed_x_axis = trimmed_x_axis
        ds_trimmed_y_axis = trimmed_y_axis
        ds_trimmed_image = trimmed_image

    plt.plot(ds_trimmed_x_axis)
    plt.plot(ds_trimmed_y_axis)
    plt.show()

    plt.pcolormesh(
        ds_trimmed_x_axis,
        ds_trimmed_y_axis,
        np.nanmax(ds_trimmed_image, axis=0),
    )
    plt.axis("scaled")
    plt.xlim(ds_trimmed_x_axis.max(), ds_trimmed_y_axis.min())
    plt.show()

    # ESTIMATE NOISE

    print(f"RMS = {estimate_rms(image=ds_trimmed_image, settings=settings)}")
