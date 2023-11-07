# generate.py
# Written by Thomas Hilder

"""
TODO: Add description
"""

from .settings import Settings
from .io import previous_results_exist

from astropy.io import fits
from skimage.measure import block_reduce

from copy import copy

import numpy as np
import matplotlib.pyplot as plt

def generate(settings: Settings) -> None:
    
    # INITIALISATION
    
    # Check if previous results exist
    previous_results_exist(settings=settings)
    
    # READ DATA

    # read cube data
    with fits.open(name=settings.file) as hdul:
        image = hdul[0].data
        
        nx = hdul[0].header["NAXIS1"]
        ny = hdul[0].header["NAXIS2"]
        
        cx = hdul[0].header["CRPIX1"]
        cy = hdul[0].header["CRPIX2"]
        
        pixelscale = hdul[0].header['CDELT2'] * 3600
        
        x_axis = -(np.arange(1, nx+1) - cx) * pixelscale
        y_axis = (np.arange(1, ny+1) - cy) * pixelscale
    
    plt.contourf(x_axis, y_axis, np.nanmax(image, axis=0), levels=100)
    plt.axis('scaled')
    plt.xlim(x_axis.max(), x_axis.min())
    plt.show()
    
    # ESTIMATE NOISE
    
    N = settings.DEFAULT_NCHANNELS_NOISE
    
    data = image
    x1, x2 = np.percentile(np.arange(data.shape[2]), [25, 75])
    y1, y2 = np.percentile(np.arange(data.shape[1]), [25, 75])
    x1, x2, y1, y2, N = int(x1), int(x2), int(y1), int(y2), int(N)
    RMS = np.nanstd([data[:N, y1:y2, x1:x2], data[-N:, y1:y2, x1:x2]])
    print(f"RMS = {RMS}")
    
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
    
    trimmed_image = image[:,i_y_min:i_y_max,i_x_min:i_x_max]
    
    print(i_y_min, i_y_max)
    print(i_x_min, i_x_max)
    
    trimmed_x_axis = x_axis[i_x_min:i_x_max]
    trimmed_y_axis = y_axis[i_y_min:i_y_max]
    
    # DOWNSAMPLE
    
    BLOCK_SIZE = settings.downsample
    
    if BLOCK_SIZE > 1:
        
        ds_trimmed_shape = block_reduce(trimmed_image[0,:,:], block_size=BLOCK_SIZE, func=np.mean).shape
        
        ds_trimmed_image = np.zeros((image.shape[0], *ds_trimmed_shape))
        
        for i in range(ds_trimmed_image.shape[0]):
            ds_trimmed_image[i,:,:] = block_reduce(trimmed_image[i,:,:], block_size=BLOCK_SIZE, func=np.mean)
            
        ds_trimmed_x_axis = block_reduce(trimmed_x_axis, block_size=BLOCK_SIZE, func=np.mean)
        ds_trimmed_y_axis = block_reduce(trimmed_y_axis, block_size=BLOCK_SIZE, func=np.mean)
        
        ds_trimmed_x_axis[-1] = ds_trimmed_x_axis[-2] + (ds_trimmed_x_axis[-2] - ds_trimmed_x_axis[-3])
        ds_trimmed_y_axis[-1] = ds_trimmed_y_axis[-2] + (ds_trimmed_y_axis[-2] - ds_trimmed_y_axis[-3])
        
    else:
        
        ds_trimmed_x_axis = trimmed_x_axis
        ds_trimmed_y_axis = trimmed_y_axis
        ds_trimmed_image = trimmed_image
    
    plt.contourf(ds_trimmed_x_axis, ds_trimmed_y_axis, np.nanmax(ds_trimmed_image, axis=0), levels=100)
    plt.axis('scaled')
    plt.xlim(ds_trimmed_x_axis.max(), ds_trimmed_y_axis.min())
    plt.show()
    
    # GET BOUNDS ON PRIORS
    
    