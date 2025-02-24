# data_covariances.py
# Thomas Hilder

import numpy as np
import pylops as pl
from numpy.typing import NDArray

from .convolution_matrix import H_operator


def C_operator(
    rms: float,
    n_x: int,
    n_y: int,
    kernel_array: NDArray,
    approx: bool = False,
) -> pl.LinearOperator:
    if approx:
        return rms**2 * pl.Identity(n_x * n_y)
    # Beam convolution sets the correlation
    H = H_operator(n_x, n_y, kernel_array)
    # Get covariances
    return rms**2 * H * (1 / kernel_array.max())
