# models.py
# Writen by Thomas Hilder

"""
TODO: Add description.
"""

from pathlib import Path

import numpy as np
from cmdstanpy import CmdStanModel
from numpy.typing import NDArray
from pkg_resources import resource_filename

# List containing valid model names (maps int -> model)
MODEL_NAMES = [
    "gaussian",
    "doublegaussian",
]


# List containing model .stan file locations (maps int -> model.stan)
STAN_FILES = [
    "stan_models/single_gaussian.stan",
    "stan_models/double_gaussian.stan",
]


# Function to compile model and return object based on choice
def get_model(model_number: int) -> CmdStanModel:
    # Check model name is valid
    if model_number not in range(1, len(MODEL_NAMES) + 1):
        raise KeyError(f"{model_number} is not a valid model.")
    # Get stan file corresponding to chosen model
    stan_file = Path(resource_filename("hmcmoments", STAN_FILES[model_number - 1]))
    # Get model object and compile .stan program if need be
    return CmdStanModel(stan_file=stan_file)


# Assemble data into format required by model
def format_data(line: NDArray, v_axis: NDArray, rms: float) -> dict:
    # Number of data points
    n_points = len(line)
    # Get channel spacing
    v_spacing = np.diff(v_axis).mean()
    # Get velocity range
    v_range = v_axis.max() - v_axis.min()
    return dict(
        N=n_points,
        x=v_axis,
        y=line,
        u_y=np.repeat(rms, n_points),
        a_lower=5*rms,
        a_upper=1.5*line.max(),
        b_lower=v_axis.min(),
        b_upper=v_axis.max(),
        c_lower=v_spacing,
        c_upper=0.1*v_range,
    )