# generate.py
# Written by Thomas Hilder

"""
TODO: Add description
"""

from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from skimage.measure import block_reduce

from .image import estimate_rms
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

    peak_intensity = np.max(np.nan_to_num(image), axis=0)

    summed_y = np.max(peak_intensity, axis=1)
    summed_x = np.max(peak_intensity, axis=0)

    threshold_x = 1.5 * np.median(summed_y)
    threshold_y = 1.5 * np.median(summed_x)

    i_x_min = np.argmin(summed_x < threshold_x) - 10
    i_y_min = copy(i_x_min)

    i_x_max = image.shape[1] - i_x_min
    i_y_max = copy(i_x_max)

    trimmed_image = image[:, i_y_min:i_y_max, i_x_min:i_x_max]

    print(i_y_min, i_y_max)
    print(i_x_min, i_x_max)

    trimmed_x_axis = x_axis[i_x_min:i_x_max]
    trimmed_y_axis = y_axis[i_y_min:i_y_max]

    # DOWNSAMPLE

    BLOCK_SIZE = settings.downsample

    if BLOCK_SIZE > 1:
        ds_trimmed_shape = block_reduce(
            trimmed_image[0, :, :], block_size=BLOCK_SIZE, func=np.mean
        ).shape

        ds_trimmed_image = np.zeros((image.shape[0], *ds_trimmed_shape))

        for i in range(ds_trimmed_image.shape[0]):
            ds_trimmed_image[i, :, :] = block_reduce(
                trimmed_image[i, :, :], block_size=BLOCK_SIZE, func=np.mean
            )

        ds_trimmed_x_axis = block_reduce(
            trimmed_x_axis, block_size=BLOCK_SIZE, func=np.mean
        )
        ds_trimmed_y_axis = block_reduce(
            trimmed_y_axis, block_size=BLOCK_SIZE, func=np.mean
        )

        ds_trimmed_x_axis[-1] = ds_trimmed_x_axis[-2] + (
            ds_trimmed_x_axis[-2] - ds_trimmed_x_axis[-3]
        )
        ds_trimmed_y_axis[-1] = ds_trimmed_y_axis[-2] + (
            ds_trimmed_y_axis[-2] - ds_trimmed_y_axis[-3]
        )

    else:
        ds_trimmed_x_axis = trimmed_x_axis
        ds_trimmed_y_axis = trimmed_y_axis
        ds_trimmed_image = trimmed_image

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
