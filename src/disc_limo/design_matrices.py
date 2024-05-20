# design_matrices.py
# Thomas Hilder, adapted from code by Hogg & Villar (2021) https://arxiv.org/abs/2101.07256

import numpy as np
from numpy.typing import NDArray

from .convolution_matrix import get_H

# Constants
DELTA_OMEGA = 0.5 * np.pi  # Frequency spacing for Fourier basis functions


def get_data_points(n: int) -> NDArray[np.float64]:
    return np.arange(0.5 / n, 1.0, 1.0 / n)


def fourier_design_matrix(
    n: int, p: int, delta_omega: float = DELTA_OMEGA
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create 1D Fourier design matrix."""
    # Get t values
    t_vals = get_data_points(n)
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


def fourier_design_matrix_2D(
    n_x: int, n_y: int, n_fourier_x: int, n_fourier_y: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create 2D Fourier design matrix as Kronecker product of two 1D matrices."""
    # 1D Fourier design matrices
    fourier_design_x, freqs_x = fourier_design_matrix(n_x, n_fourier_x)
    fourier_design_y, freqs_y = fourier_design_matrix(n_y, n_fourier_y)

    # Create 2D Fourier design matrices
    fourier_design_2D = np.kron(fourier_design_x, fourier_design_y)
    freqs_2D = np.sqrt(np.add.outer(freqs_x**2, freqs_y**2))
    freqs_2D_vector = freqs_2D.flatten()

    return fourier_design_2D, freqs_2D_vector


def full_design_and_convolution_matrices(
    n_x: int, n_y: int, n_fourier: int, kernel_array: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Create full design matrix including convolution with the beam. This is calculated as
    A = H @ Axy
    where A is the full design matrix, H is the convolution matrix and Axy is the 2D Fourier
    design matrix calculated as the Kronecker product of two 1D Fourier design matrices.
    """
    # Get convolution matrix
    convolution_matrix = get_H(n_x, n_y, kernel_array)

    # Fourier Design matrix, and frequencies of Fourier modes for feature weighting
    fourier_design_2D, freqs_2D_vector = fourier_design_matrix_2D(
        n_x, n_y, n_fourier, n_fourier
    )

    # Include convolution in full design matrix
    design_matrix = convolution_matrix @ fourier_design_2D

    return design_matrix, freqs_2D_vector, convolution_matrix
