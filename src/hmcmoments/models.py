# models.py
# Writen by Thomas Hilder

"""
TODO: Add description.
"""

from pathlib import Path

from cmdstanpy import CmdStanModel
from pkg_resources import resource_filename
from numpy.typing import NDArray

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
def get_data(image: NDArray, v_axis: NDArray):
    pass
