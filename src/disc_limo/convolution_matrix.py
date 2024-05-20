# convolution_matrix.py
# Thomas Hilder

import numpy as np
from numpy.typing import NDArray


def get_cut_and_padded_kernel(
    kernel_array: NDArray[np.float64], image_shape: tuple, i_pix: int, j_pix: int
) -> NDArray[np.float64]:
    """
    Useful function for putting together H by cutting down/padding
    the kernel.
    """
    up = i_pix - kernel_array.shape[0] // 2
    down = (image_shape[0] - i_pix) - kernel_array.shape[0] // 2 - 1
    left = j_pix - kernel_array.shape[1] // 2
    right = (image_shape[1] - j_pix) - kernel_array.shape[1] // 2 - 1
    if up < 0:
        kernel_array = kernel_array[-up:, :]
        up = 0
    if down < 0:
        kernel_array = kernel_array[:down, :]
        down = 0
    if left < 0:
        kernel_array = kernel_array[:, -left:]
        left = 0
    if right < 0:
        kernel_array = kernel_array[:, :right]
        right = 0
    res = np.pad(
        array=kernel_array,
        pad_width=[
            (up, down),
            (left, right),
        ],
        mode="constant",
        constant_values=0,
    )
    return res


def get_H(n_x: int, n_y: int, kernel_array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Function to put together the convolution matrix H."""
    H = np.zeros((n_x * n_y, n_x * n_y))
    for i in range(n_x * n_y):
        k = i // n_x
        l = i % n_y
        H[i, :] = get_cut_and_padded_kernel(kernel_array, (n_y, n_x), k, l).flatten()
    return H


def get_H_sparse_entries(
    n_x: int, n_y: int, kernel_array: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], int, int]:
    """Function to put together the convolution matrix H as a sparse matrix."""
    M = n_x * n_y
    N = n_x * n_y
    data: list = []
    row_ind: list = []
    col_ind: list = []

    for i in range(n_x * n_y):
        k = i // n_x
        l = i % n_y
        v = get_cut_and_padded_kernel(kernel_array, (n_y, n_x), k, l).flatten()

        is_non_zero = v > 0
        row_ind.extend([i] * int(np.sum(is_non_zero)))
        col_ind.extend(np.where(is_non_zero)[0])
        data.extend(v[is_non_zero])

    return np.array(data), np.array(row_ind), np.array(col_ind), M, N
