# training.py
# Thomas Hilder, adapted from code by Hogg & Villar (2021) https://arxiv.org/abs/2101.07256

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fit_channels import Setup

import numpy as np
from numpy.typing import NDArray

# Constants
RCOND = 1e-16  # Cut-off ratio for small singular values of A in solve using NumPy's least squares function


def solve(A: NDArray[np.float64], Y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve AX=Y using least squares, wrapper of numpy"""
    result, _, _, _ = np.linalg.lstsq(A, Y, rcond=RCOND)
    return result


def train_feature_weighted_gls(
    data_vector: NDArray[np.float64], fit_info: Setup
) -> NDArray[np.float64]:
    """Solve for weights vector given data vector and other pre-calculated matrices."""
    # Rename variables for readability
    A = fit_info.full_design
    Y = data_vector
    AT_Cinv = fit_info.AT_Cinv
    AT_Cinv_A_plus_L = fit_info.AT_Cinv_A  # This already has Lambda added
    Linv_AT = fit_info.Linv_AT
    A_Linv_AT_plus_C = fit_info.A_Linv_AT  # This already has C added
    # Get number of data points n and number of parameters p
    n, p = A.shape
    # Solve for X in that case that p < n
    if p < n:
        # Solve X = (A.T @ C^-1 @ A + Lambda)^-1 @ A.T @ C^-1 @ Y
        X = solve(AT_Cinv_A_plus_L, AT_Cinv @ Y)
    # Solve for X in that case that p >= n
    else:
        # Solve X = Lambda^-1 @ A.T @ (A @ Lambda^-1 @ A.T + C)^-1 @ Y
        X = Linv_AT @ solve(A_Linv_AT_plus_C, Y)
    return X


def weight_function_exp(
    omegas: NDArray[np.float64], width: float
) -> NDArray[np.float64]:
    """
    Weights function for feature weighting of Fourier design matrix, gives exponential
    kernel in limit of infinite features.
    """
    return np.asarray(
        np.sqrt(np.sqrt(2.0 / np.pi) / (1.0 / width + width * omegas**2)), np.float64
    )


def weight_function_mat32(
    omegas: NDArray[np.float64], width: float
) -> NDArray[np.float64]:
    """
    Weights function for feature weighting of Fourier design matrix, gives Matern-3/2
    kernel in limit of infinite features.
    """
    return np.asarray(1.0 / (width**2 * omegas**2 + 1.0), np.float64)
