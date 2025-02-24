# setup.py
# Thomas Hilder

import json
from collections import namedtuple

from astropy.convolution import Gaussian2DKernel

from .data_covariances import C_operator
from .design_matrices import design_operators

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
    approx_C: bool,
) -> Setup:
    """
    Calculate all operators etc. that are not changed between either fits of different
    channels/images or different CV steps (so we don't build Λ).
    """
    # For now we are not handling rectangular images
    if n_x != n_y:
        raise NotImplementedError("Only square images supported currently.")
    # Get design operator, Fourier operator, frequencies, conv operator
    F, A, ω, H = design_operators(n_x, n_y, n_fourier_x, n_fourier_y, beam_kernel.array)
    # Get data covariances operator
    C = C_operator(rms, n_x, n_y, beam_kernel.array, approx_C)
    # Store operators and hyperparameters in named tuple
    return Setup(A=A, F=F, H=H, C=C, λ=λ, ω=ω, s=s)


def save_setup(setup: Setup, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(setup._asdict(), f)


def load_setup(filename: str) -> Setup:
    """
    Reads fit setup from setup.json file, used for drawing samples and/or converting
    best-fit/samples to images.
    """
    with open(filename, "r") as f:
        data = json.load(f)
    return Setup(**data)
