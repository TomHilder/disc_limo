# design_matrices.py
# Thomas Hilder, adapted from code by Hogg & Villar (2021) https://arxiv.org/abs/2101.07256

import numpy as np
from numpy.typing import NDArray

# Constants
DELTA_OMEGA = 0.5 * np.pi  # Frequency spacing for Fourier basis functions


def fourier_design_matrix(
    t_vals: NDArray[np.float64], p: int, delta_omega: float = DELTA_OMEGA
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create 1D Fourier design matrix."""
    # Initialise arrays
    omegas = np.zeros(p) + np.nan
    design_matrix = np.zeros((len(np.atleast_1d(t_vals)), p))
    # Set zeroth values
    omegas[0] = 0.0
    design_matrix[:, 0] = 1.0
    # Set matrix entries following Hogg & Villar (2021)
    for j in range(1, p):
        omega = np.floor((j + 1.0001) / 2.0) * delta_omega
        omegas[j] = omega
        if j % 2 == 1:
            design_matrix[:, j] = np.sin(omega * t_vals)
        else:
            design_matrix[:, j] = np.cos(omega * t_vals)
    return design_matrix, omegas
