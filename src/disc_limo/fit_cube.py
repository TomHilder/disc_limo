# fit_channels.py
# Thomas Hilder

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .cube_io import read_cube, upsampled_beam
from .preconditioner import M_operator
from .regularisation import Λ_operator
from .setup import Setup, setup_fit
from .training import train_feature_weighted_gls

# TODO: replace with functions that make linear operators if need be
#       fix variable names (use greek letters)


BLOCKSIZE_MULT = 4


def fit_many_channels(
    image: NDArray[np.float64],
    channel_indicies: NDArray[np.int64],
    fit_info: Setup,
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
    results_name: str,
    filename: str,
    n_pix: int,
    n_fourier: int,
    weighting_width_inverse: float,
    lambda_coefficient: float,
    approximate_data_cov: bool = False,
    plotting: bool = False,
) -> None:
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

    # Retrieve operators
    fit_info = setup_fit(
        n_x=n_x,
        n_y=n_y,
        beam_kernel=beam,
        rms=rms,
        n_fourier_x=n_fourier,
        n_fourier_y=n_fourier,
        s=weighting_width_inverse,
        λ=lambda_coefficient,
        approx_C=approximate_data_cov,
    )
    # Calculate a preconditioner for solves with full covariance matrix
    if approximate_data_cov:
        M = None
    else:
        print("Bulding a preconditioner:")
        M = M_operator(fit_info, block_size=BLOCKSIZE_MULT * n_x)

    # # Fit all channels
    # print("Calculating posterior means of Fourier weights for each channel:")
    # weights_vectors = fit_many_channels(image, np.arange(n_channels), fit_info)
    # print("Fit complete!")
    # return weights_vectors, fit_info.weights_covariances


# def get_design_matrices(
#     filename: str,
#     n_pix: int,
#     n_fourier: int,
#     n_eval: Optional[int] = None,
# ):
#     """
#     TODO: Docstring! This function is user-accessible!
#     """
#     # n_eval = n_pix if not provided by user
#     n_eval = n_pix if n_eval is None else n_eval
#     # Read the cube to get the header only
#     _, header, *_ = read_cube(filename)
#     # Get the beam kernel evaluated at correct scale for n_eval points
#     beam = upsampled_beam(header, n_pix, n_eval)
#     # Get the design matrices
#     fourier_design, full_design, *_ = design_and_convolution_matrices(
#         n_eval, n_eval, n_fourier, beam.array
#     )
#     return fourier_design, full_design
