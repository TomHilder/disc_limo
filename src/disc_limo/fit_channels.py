# fit_channels.py
# Thomas Hilder


from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import lu
from tqdm import tqdm

from .convolution_matrix import get_H_sparse_entries
from .cube_io import estimate_rms, read_beam, read_nspaxels
from .design_matrices import fourier_design_matrix
from .training import (
    solve,
    train_feature_weighted_gls,
    weight_function_exp,
    weight_function_mat32,
)

# Named tuple for saving calculated matrices for re-use in fitting
Setup = namedtuple(
    "Setup",
    (
        "convolution_filter "
        "fourier_design "
        "full_design "
        "frequencies "
        "lambda_diagonal "
        "data_covariances "
        "weights_covariances "
        "AT_Cinv "
        "AT_Cinv_A "
        "Linv_AT "
        "A_Linv_AT"
    ),
)


def setup_fit(
    n_x: int,
    n_y: int,
    beam_kernel: Gaussian2DKernel,
    rms: float,
    n_fourier: int,
    weighting_width_inverse: float,
    lambda_coefficient: float,
) -> Setup:
    """
    Calculate everything needed for the fit and return named tuple containing quanities we want
    to avoid re-calcualating since they are constant for all channels (for example the variances
    on the best fits).
    """

    # Get convolution matrix
    # TODO: create get_convolution matrix function and move to design_matrices.py to allow for in-place re-creation
    values, row_indicies, column_indicies, *_ = get_H_sparse_entries(
        n_x, n_y, beam_kernel.array
    )
    convolution_matrix = sparse.csr_array(
        (values, (row_indicies, column_indicies)), shape=(n_x * n_y, n_x * n_y)
    )
    convolution_matrix = convolution_matrix.todense()

    # 1D Fourier design matrices
    fourier_design_x, freqs_x = fourier_design_matrix(n_x, n_fourier)
    fourier_design_y, freqs_y = fourier_design_matrix(n_y, n_fourier)

    # Create 2D Fourier design matrices
    fourier_design_2D = np.kron(fourier_design_x, fourier_design_y)
    freqs_2D = np.sqrt(np.add.outer(freqs_x**2, freqs_y**2))
    freqs_2D_vector = freqs_2D.flatten()

    # Include convolution in full design matrix
    design = convolution_matrix @ fourier_design_2D

    # Values along main diagonal of regularisation weighting matrix lambda
    lambda_diagonal = (
        lambda_coefficient
        / weight_function_exp(freqs_2D_vector, weighting_width_inverse) ** 2
    )

    # Covariance matrix
    # TODO: move to training.py in a function
    data_covariances = rms**2 * convolution_matrix / convolution_matrix.max()

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
    AT_Cinv_A[np.diag_indices(n_fourier**2)] += lambda_diagonal
    # Invert using an PLU decomposition
    permutation, l_factor, u_factor = lu(AT_Cinv_A)
    # First solve L @ y = P.T @ I
    # y = np.linalg.lstsq(l_factor, permutation.T, rcond=RCOND)[0]
    y = solve(l_factor, permutation.T)
    # Now solve U @ x = y which gives us the inverse of A.T @ C^-1 @ A + Lambda
    # weights_covariances = np.linalg.lstsq(u_factor, y, rcond=RCOND)[0]
    weights_covariances = solve(u_factor, y)

    # Some other matrices to avoid recalculating when fitting
    # L^-1 @ A.T = (A @ L^-1.T).T = (A @ L^-1).T since L is diagonal
    Linv_AT = (design / lambda_diagonal).T
    A_Linv_AT = design @ Linv_AT + data_covariances

    # Return needed quantities in named tuple
    return Setup(
        convolution_filter=convolution_matrix,
        fourier_design=fourier_design_2D,
        full_design=design,
        frequencies=freqs_2D_vector,
        lambda_diagonal=lambda_diagonal,
        data_covariances=data_covariances,
        weights_covariances=weights_covariances,
        AT_Cinv=AT_Cinv,
        AT_Cinv_A=AT_Cinv_A,
        Linv_AT=Linv_AT,
        A_Linv_AT=A_Linv_AT,
    )


def fit_single_channel(
    data_vector: NDArray[np.float64],
    fit_info: Setup,
) -> NDArray[np.float64]:
    # Train model
    return train_feature_weighted_gls(data_vector, fit_info)


def fit_many_channels(
    image: NDArray[np.float64], channel_indicies: NDArray[np.int64], fit_info: Setup
) -> NDArray[np.float64]:
    weight_vectors = []
    # Fit for each specified channel and append results
    for i in tqdm(channel_indicies):
        weight_vector = fit_single_channel(
            data_vector=image[i, :, :].flatten().T, fit_info=fit_info
        )
        weight_vectors.append(weight_vector)
    return np.array(weight_vectors)


def fit_cube(
    filename: str,
    n_pix: int,
    n_fourier: int,
    weighting_width_inverse: float,
    lambda_coefficient: float,
    plotting: bool = False,
) -> tuple[NDArray[np.float64], Setup]:
    """
    TODO: Docstring! This function is user-accessible!
    """

    # Read the cube
    cube = fits.open(filename)
    image, header = cube[0].data, cube[0].header
    n_x_total, n_y_total, n_channels = read_nspaxels(header)
    beam = Gaussian2DKernel(*read_beam(header, 1))
    rms = estimate_rms(image)

    if plotting:
        plt.imshow(beam.array)
        plt.show()

    # Trim cube TODO: move to own function
    assert n_x_total == n_y_total
    centre_index = n_x_total // 2
    lower_index = centre_index - (n_pix // 2)
    upper_index = centre_index + (n_pix // 2)
    trimmed_image = image[:, lower_index:upper_index, lower_index:upper_index]
    del image
    n_x, n_y = trimmed_image[0, :, :].shape

    if plotting:
        plt.imshow(trimmed_image[n_channels // 2, :, :])
        plt.show()

    fit_info = setup_fit(
        n_x,
        n_y,
        beam,
        rms,
        n_fourier,
        weighting_width_inverse,
        lambda_coefficient,
    )

    if plotting:
        vmax = float(np.percentile(fit_info.weights_covariances, 99.9))
        plt.imshow(fit_info.weights_covariances, cmap="RdBu", vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.show()

    weight_vectors = fit_many_channels(trimmed_image, np.arange(n_channels), fit_info)

    return weight_vectors, fit_info
