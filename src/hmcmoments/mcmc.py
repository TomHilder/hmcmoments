# mcmc.py
# Written by Thomas Hilder

"""
Perform Hamiltonian Monte Carlo inference on every line profile in a cube using the CMDStanPy interface to Stan
"""

import logging
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm
from functools import partial
from cmdstanpy import CmdStanModel

from .image import estimate_rms
from .models import format_data, get_model
from .settings import Settings


def do_mcmc(image: NDArray, v_axis: NDArray, settings: Settings):
    # Get CMDStanModel object containing our model info
    model = get_model(settings.model)
    # Estimate rms
    rms = estimate_rms(image, settings)
    
    
    results = np.zeros((*image.shape[1:], 4, 9))
    
    for i in tqdm(range(image.shape[1])):
        
        i_args = dict(
            i=i,
            v_axis=v_axis,
            image=image,
            rms=rms,
            model=model,
        )
        
        get_stats_partial = partial(get_statistics_mcmc, **i_args)
            
        with multiprocessing.get_context("spawn").Pool(processes=5) as pool:
            results[i,:,:,:] = pool.map(get_stats_partial, range(image.shape[2]))
    
    plt.imshow(np.nanmax(image, axis=0), origin="lower")
    plt.show()
    plt.imshow(results[:,:,2,0], origin="lower", cmap="RdBu")
    plt.colorbar()
    plt.show()
    
    
def get_statistics_mcmc(j: int, i: int, v_axis: NDArray, image: NDArray, rms: float, model: CmdStanModel) -> NDArray:
    
    # Disable logging printing too much shit
    logger = logging.getLogger("cmdstanpy")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    
    line = image[:,i,j]
            
    data = format_data(line, v_axis, rms)
    
    initialisation = dict(
        a=line.max(),
        b=v_axis[np.argmax(line)],
        c=1e3,
    )
    
    try:
        fit = model.sample(
            data=data,
            chains=2,
            parallel_chains=2,
            show_progress=False,
            show_console=False,
            iter_warmup=1000,
            iter_sampling=2000,
            max_treedepth=12,
            adapt_delta=0.99,
            inits=initialisation,
            refresh=1000,
        )
    
        return fit.summary()
        
    except RuntimeError:
        
        return np.zeros((4,9))