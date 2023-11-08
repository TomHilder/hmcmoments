# mcmc.py
# Written by Thomas Hilder

"""
Perform Hamiltonian Monte Carlo on the line profiles in a cube using the CMDStanPy interface to Stan
"""

from numpy.typing import NDArray

from .models import get_model
from .settings import Settings


def do_mcmc(image: NDArray, settings: Settings):
    # Get CMDStanModel object containing our model info
    model = get_model(settings.model)
    # Assemble data dictionary
