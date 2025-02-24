# convolution_matrix.py
# Thomas Hilder

import pylops as pl
from numpy.typing import NDArray

from .dtype import FLOAT_DTYPE


def H_operator(n_x: int, n_y: int, kernel_array: NDArray) -> pl.LinearOperator:
    """Generate a linear operator representing the convolution matrix H."""
    return pl.signalprocessing.Convolve2D(
        dims=(n_x, n_y),
        h=kernel_array,
        offset=(kernel_array.shape[0] // 2, kernel_array.shape[1] // 2),
        method="fft",
        dtype=FLOAT_DTYPE,
    )
