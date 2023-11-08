# image.py
# Written by Thomas Hilder

"""
Functions to manipulate cube image data.
"""

import numpy as np
from numpy.typing import NDArray

from .settings import Settings


# Estimate the noise in a cube
def estimate_rms(image: NDArray, settings: Settings) -> float:
    return float(
        np.std(
            a=np.concatenate(
                [
                    image[: settings.DEFAULT_NCHANNELS_NOISE, :, :],
                    image[-settings.DEFAULT_NCHANNELS_NOISE :, :, :],
                ]
            )
        )
    )


# Trim cube image down to region containing disc
def trim_cube(
    x_axis: NDArray, y_axis: NDArray, image: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    pass


# Downsample cube image
def downsample_cube(
    x_axis: NDArray, y_axis: NDArray, image: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    pass
