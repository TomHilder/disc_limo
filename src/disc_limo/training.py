# training.py
# Thomas Hilder, adapted from code by Hogg & Villar (2021) available at
# https://arxiv.org/abs/2101.07256

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fit_channels import Setup

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu

# Constants
# Cut-off ratio for small singular values of A in solve using NumPy's least squares
# function
RCOND = 1e-16


def solve(A: NDArray[np.float64], Y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve AX=Y using least squares, wrapper of numpy"""
    result, _, _, _ = np.linalg.lstsq(A, Y, rcond=RCOND)
    return result


def calc_weight_covariances_and_matrices(
    design: NDArray[np.float64],
    freqs_2D_vector: NDArray[np.float64],
    convolution: NDArray[np.float64],
    rms: float,
    lambda_coefficient: float,
    weighting_width_inverse: float,
):
    """
    Calculates the covariances on the weights in every channel (they are the same in all
    channels) and a bunch of matrices that are constant across fits for each channel to
    save (a lot) of time.
    """
    # Values along main diagonal of regularisation weighting matrix lambda
    lambda_diagonal = (
        lambda_coefficient
        / weight_function_exp(freqs_2D_vector, weighting_width_inverse) ** 2
    )
    # Data covariance matrix from correlation (convolution) matrix
    # This is a short-cut that is equivalent to the full expression for
    # the covariance matrix in terms of the correlation matrix since the
    # scale of the noise in all pixels is assumed constant.
    data_covariances = rms**2 * convolution / convolution.max()
    # ## Calculate variances on weights for fits
    # First calculate A.T @ C^-1 @ A + Lambda
    # Setting Gamma = A.T @ C^-1 implies Gamma @ C = A.T
    # Tansposing both sides gives C.T @ Gamma.T = A
    # Solving that with lst sqrs then transposing gives us Gamma aka A.T @ C^-1
    # AT_Cinv_A = design.T @ np.linalg.lstsq(data_covariances, design, rcond=RCOND)[0]
    # AT_Cinv = np.linalg.lstsq(data_covariances.T, design, rcond=RCOND)[0].T
    AT_Cinv = solve(data_covariances.T, design).T
    AT_Cinv_A = AT_Cinv @ design
    # Add feature weighting constraint
    n_fourier = int(np.sqrt(design.shape[1]))
    AT_Cinv_A[np.diag_indices(n_fourier**2)] += lambda_diagonal
    # Invert using an PLU decomposition
    permutation, l_factor, u_factor = lu(AT_Cinv_A)
    # First solve L @ y = P.T @ I
    y = solve(l_factor, permutation.T)
    # Now solve U @ x = y which gives us the inverse of A.T @ C^-1 @ A + Lambda
    weights_covariances = solve(u_factor, y)
    # Some other matrices to avoid recalculating when fitting
    # L^-1 @ A.T = (A @ L^-1.T).T = (A @ L^-1).T since L is diagonal
    Linv_AT = (design / lambda_diagonal).T
    A_Linv_AT = design @ Linv_AT + data_covariances
    return weights_covariances, AT_Cinv, AT_Cinv_A, Linv_AT, A_Linv_AT


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
