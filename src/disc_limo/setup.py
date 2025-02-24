# setup.py
# Thomas Hilder

from collections import namedtuple

import numpy as np
from numpy.typing import NDArray

from .data_covariances import C_operator
from .design import design_operators

# Named tuple for operators, frequencies vector and hyperparameters
Setup = namedtuple("Setup", ["A", "F", "H", "C", "Cinv", "λ", "ω", "s", "savedata"])


def setup_fit(
    n_x: int,
    n_y: int,
    beam_kernel_array: NDArray | list,
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
    # Ensure beam_kernel_array is NDArray
    beam_kernel_array = np.array(beam_kernel_array)
    # Keep args so we can save them later
    savedata = {
        "n_x": n_x,
        "n_y": n_y,
        "beam_kernel_array": beam_kernel_array.tolist(),
        "rms": rms,
        "n_fourier_x": n_fourier_x,
        "n_fourier_y": n_fourier_y,
        "s": s,
        "λ": λ,
        "approx_C": approx_C,
    }
    # For now we are not handling rectangular images
    if n_x != n_y:
        raise NotImplementedError("Only square images supported currently.")
    # Get design operator, Fourier operator, frequencies, conv operator
    F, A, ω, H = design_operators(n_x, n_y, n_fourier_x, n_fourier_y, beam_kernel_array)
    # Get data covariances operator
    C, Cinv = C_operator(rms, n_x, n_y, beam_kernel_array, approx_C)
    # Store operators and hyperparameters in named tuple
    return Setup(A=A, F=F, H=H, C=C, Cinv=Cinv, λ=λ, ω=ω, s=s, savedata=savedata)
