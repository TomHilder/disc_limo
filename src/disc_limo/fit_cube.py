# fit_cube.py
# Thomas Hilder

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pylops as pl
from astropy.io.fits.header import Header
from numpy.typing import NDArray
from tqdm import tqdm

from .cube_io import read_cube
from .linear_solve import linear_solve
from .precision import dense_precision
from .preconditioner import M_operator
from .results_io import save_all
from .setup import Setup, setup_fit

BLOCKSIZE_MULT = 2


def fit_many_channels(
    image: NDArray[np.float64],
    channel_indicies: NDArray[np.int64],
    fit_info: Setup,
    precon: pl.LinearOperator,
) -> tuple[NDArray[np.float64], dict]:
    X_vals = []
    info_vals = []
    t_vals = []
    # Fit for each specified channel and append results
    for j in tqdm(channel_indicies, desc="fitting channels"):
        # Solve this channel
        Y = image[j, :, :].flatten().T
        X, i, t = linear_solve(fit_info, Y, precon)
        # Save outputs
        X_vals.append(X)
        info_vals.append(i)
        t_vals.append(t)
    # Convert to arrays and dict for meta
    X_results = np.array(X_vals)
    meta = {
        "info": info_vals,
        "t_solves": t_vals,
    }
    return X_results, meta


def fit_cube(
    filename: str,
    n_pix: int,
    n_fourier: int,
    weighting_width_inverse: float,
    lambda_coefficient: float,
    get_dense_precision: bool = False,
    save: Optional[str] = None,
    approximate_data_cov: bool = False,
    plotting: bool = False,
    channel_inds: Optional[list[int]] = None,
) -> tuple[NDArray, Setup, dict, Header]:
    """
    TODO: Docstring! This function is user-accessible!
    """

    # Read the cube
    image, header, beam, rms, n_x, n_y, n_channels = read_cube(filename, n_pix)
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
        beam_kernel_array=beam.array,
        rms=rms,
        n_fourier_x=n_fourier,
        n_fourier_y=n_fourier,
        s=weighting_width_inverse,
        λ=lambda_coefficient,
        approx_C=approximate_data_cov,
    )
    # Calculate a preconditioner for solves with full covariance matrix
    if approximate_data_cov:
        # if True:
        M = None
    else:
        block_size = BLOCKSIZE_MULT * n_x
        # print(f"Bulding a preconditioner:")
        M = M_operator(fit_info, block_size=block_size)

    # Fit all channels
    if channel_inds is None:
        channel_inds = np.arange(n_channels)
    results, meta = fit_many_channels(image, channel_inds, fit_info, M)
    # Get dense precision matrix
    if get_dense_precision:
        Ω = dense_precision(fit_info)
        np.save(save + "precision.npy", Ω)

    # Save results
    if save is not None:
        save_all(
            filename_base=save,
            results=results,
            setup=fit_info,
            meta=meta,
            fitsheader=header,
        )
    # And return too
    print("done!")
    return results, fit_info, meta, header


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
