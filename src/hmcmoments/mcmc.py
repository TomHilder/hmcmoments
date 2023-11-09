# mcmc.py
# Written by Thomas Hilder

"""
Perform Hamiltonian Monte Carlo inference on every line profile in a cube using the CMDStanPy interface to Stan
"""

import logging
import multiprocessing
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from cmdstanpy import CmdStanModel
from numpy.typing import NDArray
from tqdm import tqdm

from .image import estimate_rms
from .models import format_data, get_model
from .settings import Settings


class HMCSamplingConfig:
    """Class to keep track of global settings we provide to the CmdStanModel.sample method.

    This is a class instead of a dataclass because it is not intended to change between runs
    in most cases, and so it is not intended to be instantiated.
    """

    # Number of chains (which are executed always in parallel)
    N_CHAINS: int = 2
    # Number of iterations
    N_BURNIN: int = 1000
    N_PRODUCTION: int = 2000
    # Tuning sensitivity
    MAX_TREEDEPTH: int = 12
    ADAPT_DELTA: float = 0.99
    # Visibility
    SHOW_PROGRESS: bool = False
    SHOW_CONSOLE: bool = False
    REFRESH: int = 1000

    @classmethod
    def get_sampler_kwargs(cls):
        return dict(
            chains=cls.N_CHAINS,
            parallel_chains=cls.N_CHAINS,
            iter_warmup=cls.N_BURNIN,
            iter_sampling=cls.N_PRODUCTION,
            max_treedepth=cls.MAX_TREEDEPTH,
            adapt_delta=cls.ADAPT_DELTA,
            show_progress=cls.SHOW_PROGRESS,
            show_console=cls.SHOW_CONSOLE,
            refresh=cls.REFRESH,
        )


def do_mcmc_image(image: NDArray, v_axis: NDArray, settings: Settings) -> NDArray:
    # Get CMDStanModel object that we can use to perform sampling
    model = get_model(settings.model)
    # An estimate of the rms in the image will be used as the uncertainty
    rms = estimate_rms(image, settings)
    # Initialise an array of shape (n_x_pixels, n_y_pixels, n_params+1, n_statistics) to mcmc results
    summary_statistics = np.zeros((*image.shape[1:], 4, 9))
    # Iterate over the rows of the image sequentially
    for i in tqdm(range(image.shape[1])):
        # Reduce do_mcmc_line to single argument function for pool.map since other arguments are constant for fixed i
        do_mcmc_line_partial = partial(
            do_mcmc_line, **dict(v_axis=v_axis, rms=rms, model=model)
        )
        # Parallelise the mcmc over pixels in the chosen row, with N_CHAINS cores per process
        n_processes = settings.cores // HMCSamplingConfig.N_CHAINS
        with multiprocessing.get_context("spawn").Pool(processes=n_processes) as pool:
            # Convenience generator expression to get line profile for pixel_ij
            line_generator = (image[:, i, j] for j in range(image.shape[2]))
            # Get results for all pixel line profiles in row
            summary_statistics[i, :, :, :] = pool.map(
                do_mcmc_line_partial, line_generator
            )
    return summary_statistics

    # plt.imshow(np.nanmax(image, axis=0), origin="lower")
    # plt.show()
    # plt.imshow(summary_statistics[:, :, 2, 0], origin="lower", cmap="RdBu")
    # plt.colorbar()
    # plt.show()


def do_mcmc_line(
    line: NDArray, v_axis: NDArray, rms: float, model: CmdStanModel
) -> NDArray:
    # Disable deluge of logging that occurs when running many models
    silent_logging_cmdstanpy()
    # Get data in form needed to run Stan model (including setting parameter bounds)
    data = format_data(line, v_axis, rms)
    # Choose sensible initialisation for chains based on data
    initialisation = dict(a=line.max(), b=v_axis[np.argmax(line)], c=1e3)
    # Get sampler configuration kwargs from global settings
    kwargs_sampler = HMCSamplingConfig.get_sampler_kwargs()
    # Perform sampling
    try:
        fit = model.sample(data=data, inits=initialisation, **kwargs_sampler)
        # Return only the summary statistics from the fit
        return np.array(fit.summary())
    # If the pixel does not contain a peak compatible with the lower bound on the height
    # of the Gaussian, the sampler will return a RuntimeError. We interpret this as a lack
    # of a detection in this pixel and simply return zeros in the same format as the summary.
    except RuntimeError:
        return np.zeros((4, 9))


def silent_logging_cmdstanpy():
    """See: https://stackoverflow.com/questions/66667909/stop-printing-infocmdstanpystart-chain-1-infocmdstanpyfinish-chain-1"""
    logger = logging.getLogger("cmdstanpy")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    return logger
