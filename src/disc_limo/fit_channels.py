# fit_channels.py
# Thomas Hilder


from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from numpy.typing import NDArray
from tqdm import tqdm

from .cube_io import estimate_rms, read_beam, read_nspaxels
from .design_matrices import full_design_and_convolution_matrices
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
    design, freqs_2D_vector, convolution_matrix = full_design_and_convolution_matrices(
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
