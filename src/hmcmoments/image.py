# image.py
# Written by Thomas Hilder

"""
Functions to manipulate cube image data.
"""

import numpy as np
from numpy.typing import NDArray
from skimage.measure import block_reduce

from .settings import Settings


# Function for downsampling
DS_FUNC = np.mean


# Estimate the noise in a cube
def estimate_rms(image: NDArray, settings: Settings) -> float:
    # Estimate RMS using standard deviation of channels far from systemic velocity
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
    x_axis: NDArray,
    y_axis: NDArray,
    image: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    # Replace all NaNs in image with zeros
    image = np.nan_to_num(image, nan=0.0)
    # Calculate peak intensity along each spatial dimension
    peak_intensity_x = np.max(
        image,
        axis=(0, 2),
    )
    peak_intensity_y = np.max(
        image,
        axis=(0, 1),
    )
    # Calculate threshold for inclusion using median from each
    threshold = 1.5 * np.median(np.concatenate([peak_intensity_x, peak_intensity_y]))
    # Get indicies to make cuts (enforce square image by using same in each dimension)
    i_min = np.argmin(peak_intensity_x < threshold) - 1
    i_max = image.shape[1] - i_min
    # Return trimmed results
    return (
        x_axis[i_min:i_max],
        y_axis[i_min:i_max],
        image[:, i_min:i_max, i_min:i_max],
    )


# Downsample cube image in spatial dimensions
def downsample_cube(
    x_axis: NDArray,
    y_axis: NDArray,
    image: NDArray,
    block_size: int,
) -> tuple[NDArray, NDArray, NDArray]:
    # Constant arguments for downsampling
    ds_args = dict(block_size=block_size, func=DS_FUNC)
    # Downsample at one velocity to get dimensions of downsampled image
    ds_shape = block_reduce(image[0, :, :], **ds_args).shape
    # Create empty new image of correct shape to fill with downsampled data
    ds_image = np.zeros((image.shape[0], *ds_shape))
    # Loop over all velocities and downsample spatially
    for i in range(ds_image.shape[0]):
        ds_image[i, :, :] = block_reduce(image[i, :, :], **ds_args)
    # Downsample coordinates in both spatial axes
    ds_x_axis = block_reduce(x_axis, **ds_args)
    ds_y_axis = block_reduce(y_axis, **ds_args)

    # Correct last coordinate for the case where n_pix % block_size != 0 along each dim
    def correct_axis_after_ds(axis: NDArray) -> NDArray:
        axis[-1] = axis[-2] + (axis[-2] - axis[-3])
        return axis

    ds_x_axis = correct_axis_after_ds(ds_x_axis)
    ds_y_axis = correct_axis_after_ds(ds_y_axis)
    # Return downsampled results
    return (
        ds_x_axis,
        ds_y_axis,
        ds_image,
    )
