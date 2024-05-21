# fit_channels.py
# Thomas Hilder

from collections import namedtuple
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel
from numpy.typing import NDArray
from tqdm import tqdm

from .cube_io import read_cube, upsampled_beam
from .design_matrices import design_and_convolution_matrices
from .training import calc_weight_covariances_and_matrices, train_feature_weighted_gls

# Named tuple for saving calculated matrices for re-use in fitting
Setup = namedtuple(
    "Setup",
    (
        "full_design "
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

    # Get design matrix, fourier mode frequencies, convolution matrix
    _, design, freqs_2D_vector, convolution_matrix = design_and_convolution_matrices(
        n_x, n_y, n_fourier, beam_kernel.array
    )

    # Get weights covariances, and a bunch of matrices re-used in the fit of each channel
    weights_covariances, AT_Cinv, AT_Cinv_A, Linv_AT, A_Linv_AT = (
        calc_weight_covariances_and_matrices(
            design,
            freqs_2D_vector,
            convolution_matrix,
            rms,
            lambda_coefficient,
            weighting_width_inverse,
        )
    )

    # Return needed quantities in named tuple
    return Setup(
        full_design=design,
        weights_covariances=weights_covariances,
        AT_Cinv=AT_Cinv,
        AT_Cinv_A=AT_Cinv_A,
        Linv_AT=Linv_AT,
        A_Linv_AT=A_Linv_AT,
    )


def fit_many_channels(
    image: NDArray[np.float64], channel_indicies: NDArray[np.int64], fit_info: Setup
) -> NDArray[np.float64]:
    weight_vectors = []
    # Fit for each specified channel and append results
    for i in tqdm(channel_indicies):
        weight_vector = train_feature_weighted_gls(
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
    image, _, beam, rms, n_x, n_y, n_channels = read_cube(filename, n_pix)

    # Plots if requested
    if plotting:
        plt.imshow(beam.array)
        plt.show()
        plt.imshow(image[n_channels // 2, :, :])
        plt.show()

    # Calculate everything we can before fitting individual channels
    fit_info = setup_fit(
        n_x,
        n_y,
        beam,
        rms,
        n_fourier,
        weighting_width_inverse,
        lambda_coefficient,
    )

    # Plots if requested
    if plotting:
        vmax = float(np.percentile(fit_info.weights_covariances, 99.9))
        plt.imshow(fit_info.weights_covariances, cmap="RdBu", vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.show()

    # Fit all channels to get best fit weights
    weight_vectors = fit_many_channels(image, np.arange(n_channels), fit_info)

    return weight_vectors, fit_info.weights_covariances


def get_design_matrices(
    filename: str, n_pix: int, n_fourier: int, n_eval: Optional[int] = None
):
    """
    TODO: Docstring! This function is user-accessible!
    """
    # n_eval = n_pix if not provided by user
    n_eval = n_pix if n_eval is None else n_eval
    # Read the cube to get the header only
    _, header, *_ = read_cube(filename)
    # Get the beam kernel evaluated at correct scale for n_eval points
    beam = upsampled_beam(header, n_pix, n_eval)
    # Get the design matrices
    fourier_design, full_design, *_ = design_and_convolution_matrices(
        n_eval, n_eval, n_fourier, beam.array
    )
    return fourier_design, full_design
