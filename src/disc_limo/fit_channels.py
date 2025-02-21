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
from .data_covariances import C_operator
from .design_matrices import design_operators
from .regularisation import Λ_operator
from .training import train_feature_weighted_gls

# TODO: replace with functions that make linear operators if need be
#       fix variable names (use greek letters)

# Named tuple for operators, frequencies vector and hyperparameters
Setup = namedtuple("Setup", ["A", "F", "H", "C", "λ", "ω", "s"])


def setup_fit(
    n_x: int,
    n_y: int,
    beam_kernel: Gaussian2DKernel,
    rms: float,
    n_fourier_x: int,
    n_fourier_y: int,
    s: float,
    λ: float,
) -> Setup:
    """
    Calculate everything needed for the fit and return named tuple containing quanities
    we want to avoid re-calcualating since they are constant for all channels (for
    example the variances on the best fits).
    """
    # For now we are not handling rectangular images
    if n_x != n_y:
        raise NotImplementedError("Only square images supported currently.")
    # Get design operator, Fourier operator, frequencies, conv operator
    F, A, ω, H = design_operators(n_x, n_y, n_fourier_x, n_fourier_y, beam_kernel.array)
    # Get data covariances operator
    C = C_operator(rms, n_x, n_y, beam_kernel.array)
    # Store operators and hyperparameters in named tuple
    return Setup(A=A, F=F, H=H, C=C, λ=λ, ω=ω, s=s)


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
    print(
        "Calculating covariance matrix for Fourier weights and resused matrices... ",
        end="",
        flush=True,
    )
    fit_info = setup_fit(
        n_x,
        n_y,
        beam,
        rms,
        n_fourier,
        weighting_width_inverse,
        lambda_coefficient,
    )
    print("done.")
    # Plots if requested
    if plotting:
        vmax = float(np.percentile(fit_info.weights_covariances, 99.9))
        plt.imshow(fit_info.weights_covariances, cmap="RdBu", vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.show()
    # Fit all channels to get best fit weights
    print("Calculating posterior means of Fourier weights for each channel:")
    weights_vectors = fit_many_channels(image, np.arange(n_channels), fit_info)
    print("Fit complete!")
    return weights_vectors, fit_info.weights_covariances


def get_design_matrices(
    filename: str,
    n_pix: int,
    n_fourier: int,
    n_eval: Optional[int] = None,
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
