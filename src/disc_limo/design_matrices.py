# design_matrices.py
# Thomas Hilder, adapted from code by Hogg & Villar (2021) available at
# https://arxiv.org/abs/2101.07256

import numpy as np
import pylops as pl
from nifty_solve import Finufft2DRealOperator
from numpy.typing import NDArray

from disc_limo.convolution_matrix import H_operator

from .constants import FINUFFT_TOL, π
from .dtype import FLOAT_DTYPE


def get_data_points(n: int) -> NDArray[np.float64]:
    return np.linspace(-0.5 * π, 0.5 * π, n, dtype=FLOAT_DTYPE)


def F_operator(
    n_x: int,
    n_y: int,
    n_fourier_x: int,
    n_fourier_y: int,
) -> tuple[pl.LinearOperator, NDArray[np.float64]]:
    """
    Get linear operator representing 2D Fourier design matrix. Uses fiNUFFT and pylops
    as a backend, with operator implemented in nifty-solve.
    """
    # Image coordinates
    t_x, t_y = (get_data_points(n) for n in [n_x, n_y])
    t_x, t_y = [t.flatten() for t in np.meshgrid(t_x, t_y)]
    # Build operator
    F = Finufft2DRealOperator(
        x=t_x,
        y=t_y,
        n_modes=(n_fourier_x, n_fourier_y),
        eps=FINUFFT_TOL,
    )
    # Frequencies
    ω_x, ω_y = F.get_mode_freqs()
    ω = np.sqrt(ω_x**2 + ω_y**2)
    return F, ω


def design_operators(
    n_x: int,
    n_y: int,
    n_fourier_x: int,
    n_fourier_y: int,
    kernel_array: NDArray[np.float64],
) -> tuple[
    pl.LinearOperator, pl.LinearOperator, NDArray[np.float64], pl.LinearOperator
]:
    """
    Create linear operator representing design matrix for full forward model including
    convolution with idealised beam. Calculated as
    A = H @ F
    where A is the full design matrix, H is the convolution matrix and F is the 2D
    Fourier design matrix. ω is a vector containing the frequencies of each mode. All
    matrices are represented with linear operators.
    Function returns F, A, ω, H
    """
    # Convolution
    H = H_operator(n_x, n_y, kernel_array)
    # Fourier design operator, and frequencies of modes for feature weighting
    F, ω = F_operator(n_x, n_y, n_fourier_x, n_fourier_y)
    # Full forward model operator includes convolution
    A = H @ F
    return F, A, ω, H
