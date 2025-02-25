# cross_validation.py
# Thomas Hilder

from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pylops as pl
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from tqdm import tqdm

from .constants import π
from .cube_io import read_cube
from .design import get_image_coords
from .fit_cube import BLOCKSIZE_MULT
from .linear_solve import linear_solve
from .preconditioner import M_operator
from .setup import setup_fit


def R_operator(n_data: int, holdout_indices: NDArray) -> pl.Restriction:
    return pl.Restriction(n_data, np.delete(np.arange(n_data), holdout_indices))


def get_holdout_indices(
    t_x_centre: float,
    t_y_centre: float,
    loo_radius_t: float,
    t_x: NDArray[np.float64],
    t_y: NDArray[np.float64],
) -> NDArray:
    distances = np.sqrt((t_x - t_x_centre) ** 2 + (t_y - t_y_centre) ** 2)
    holdout_indices = np.where(distances <= loo_radius_t)[0]
    return holdout_indices


def loo_indices(
    n_x: int,
    n_y: int,
    n_loo: int,
    r_loo_pix: float,
    Y: NDArray,
    rms: float,
    rng: Generator,
) -> tuple[list[NDArray], list[list[float]]]:
    # Check that n_loo isn't too large
    if n_loo > n_x * n_y:
        raise ValueError("n_loo cannot exceed total number of pixels in images.")
    # Randomly sample some locations
    t_x, t_y = get_image_coords(n_x, n_y)
    loo_centre_inds = np.arange(t_x.shape[0])
    rng.shuffle(loo_centre_inds)
    # Convert loo radius to t coordinates
    r_loo_t = r_loo_pix * (t_x[1] - t_x[0])
    # Loop over centre indices to get all loo inds
    hold_inds: list[NDArray] = []
    loo_coords: list[list[float]] = []
    i = 0
    while len(hold_inds) < n_loo:
        i_centre = loo_centre_inds[i]
        if Y[i_centre] > 3 * rms:
            t_x_i = t_x[i_centre]
            t_y_i = t_y[i_centre]
            loo_coords.append([t_x_i, t_y_i])
            hold_inds.append(get_holdout_indices(t_x_i, t_y_i, r_loo_t, t_x, t_y))
        i += 1
    return hold_inds, loo_coords


def perform_cv(
    filename: str,
    i_channel: int,
    n_pix: int,
    n_fourier: int,
    s_vals: NDArray,
    λ_vals: NDArray,
    n_loo: int,
    r_loo: float,
    save: str,
    approximate_data_cov: bool = False,
    seed: Optional[int] = None,
):
    """
    r_loo is fraction of the beam.
    """
    # RNG
    rng = default_rng(seed)

    # Read the cube
    image, header, beam, rms, n_x, n_y, _ = read_cube(filename, n_pix)

    # Info that is fixed across hyperparameter combos
    fit_info_all_hyperparam = dict(
        n_x=n_x,
        n_y=n_y,
        beam_kernel_array=beam.array,
        rms=rms,
        n_fourier_x=n_fourier,
        n_fourier_y=n_fourier,
        approx_C=approximate_data_cov,
    )
    setup_from_λs = partial(setup_fit, **fit_info_all_hyperparam)

    # Image that we perform CV with
    Y = image[i_channel, :, :].flatten().T

    # We pick the LOO locations and find indices to exclude
    r_loo_pix = r_loo * beam._model.x_stddev.value
    loo_inds, loo_coords = loo_indices(n_x, n_y, n_loo, r_loo_pix, Y, rms, rng)

    # Empty arrays to store our results
    n_s = len(s_vals)
    n_λ = len(λ_vals)
    n_data = int(n_x * n_y)
    residuals_sq = np.zeros((n_s, n_λ, n_loo, n_data))
    held_mse = np.zeros((n_s, n_λ, n_loo))
    t_solve = np.zeros((n_s, n_λ, n_loo))
    info = np.zeros((n_s, n_λ, n_loo), dtype="int")

    # Loop over the hyperparameters
    print("Performing LOOCV:")
    for i, s in tqdm(enumerate(s_vals), position=0, total=n_s, desc="s"):
        for j, λ in tqdm(
            enumerate(λ_vals), position=1, leave=False, total=n_λ, desc="λ"
        ):

            # Build setup
            setup_λs = setup_from_λs(λ=λ, s=s)

            # Calculate preconditioner
            if approximate_data_cov:
                M = None
            else:
                block_size = BLOCKSIZE_MULT * n_x
                M = M_operator(setup_λs, block_size=block_size)

            # Loop over the LOO indices
            for k, inds_loo in tqdm(
                enumerate(loo_inds),
                position=2,
                leave=False,
                total=n_loo,
                desc="loo",
            ):
                # Get restriction operator
                R = R_operator(n_data, inds_loo)
                # Fit
                X, info[i, j, k], t_solve[i, j, k] = linear_solve(setup_λs, Y, M, R)

                # Predict data and evaluate error
                Y_pred = (setup_λs.A @ X).flatten()
                residuals_sq[i, j, k, :] = (Y_pred - Y) ** 2
                held_mse[i, j, k] = np.sum(residuals_sq[i, j, k, inds_loo])

    if save:
        np.savez(
            save + "_cv.npz",
            res_sq=residuals_sq,
            mse=held_mse,
            t=t_solve,
            i=info,
            loo_coords=loo_coords,
        )

    meta = {
        "info": info,
        "t_solves": t_solve,
    }
    print("done!")
    return residuals_sq, held_mse, meta, header, loo_inds, loo_coords
