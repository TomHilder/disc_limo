# fit_lines.py
# Thomas Hilder

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from tqdm import tqdm

MINIMZE_METHOD = "L-BFGS-B"
BOUNDS_GAUSSIAN = [(1e-8, np.inf), (-np.inf, np.inf), (1e-8, np.inf)]


def gaussian(
    x: NDArray,
    a: float,
    b: float,
    c: float,
) -> NDArray:
    return a * np.exp(-0.5 * (x - b) ** 2 * c**-2)


def neg_ln_likelihood_obj_func(
    theta: tuple,
    x: NDArray,
    y: NDArray,
    f: callable,
) -> float:
    return 0.5 * np.sum((y - f(x, *theta)) ** 2)


def fit_gaussian(x, y):
    initial_theta = (
        np.max(y),
        x[np.argmax(y)],
        0.5 * np.ptp(x),
    )
    fit = minimize(
        neg_ln_likelihood_obj_func,
        initial_theta,
        args=(x, y, gaussian),
        method=MINIMZE_METHOD,
        bounds=BOUNDS_GAUSSIAN,
    )
    return fit.x


def move_axes_for_iteration(
    data,
    n_front_axes,
    reverse=False,
):
    if n_front_axes == 0:
        return data
    source_indicies = np.arange(n_front_axes)
    dest_indicies = np.arange(-n_front_axes, 0)
    if not reverse:
        return np.moveaxis(data, source_indicies, dest_indicies)
    else:
        return np.moveaxis(data, dest_indicies, source_indicies)


def fit_gaussians(
    velocities,
    cube,
    cut_off=None,
):
    # Get cube dimensions, assuming last two dimensions are spatial and 3rd-to-last
    # dimension is spectral/velocities
    n_v, _, _ = cube.shape[-3:]
    # Reorder axes to give shape (n_v, n_i, n_j, ...) if needed
    n_front_axes = len(cube.shape) - 3
    cube = move_axes_for_iteration(cube, n_front_axes)
    # Now reduce shape to (n_v, ...) for easy iteration
    old_cube_shape = cube.shape
    flat_cube = cube.reshape((n_v, -1))

    # Iterate over all pixels in cube
    print("Fitting Gaussian line profiles to all provided pixels:")
    peak = np.zeros((flat_cube.shape[1]))
    centroid = np.zeros((flat_cube.shape[1]))
    width = np.zeros((flat_cube.shape[1]))
    for i in tqdm(range(flat_cube.shape[1])):
        # Find best fit parameters of Gaussian
        # if cut_off is not None and np.max(flat_cube[:, i]) > cut_off:
        if cut_off is None or np.max(flat_cube[:, i]) >= cut_off:
            peak[i], centroid[i], width[i] = fit_gaussian(velocities, flat_cube[:, i])
        else:
            peak[i], centroid[i], width[i] = np.nan, np.nan, np.nan

    # Return best fit parameters for each pixel as 3 arrays
    fix_shape = lambda x: move_axes_for_iteration(
        x.reshape(old_cube_shape[1:]), n_front_axes, reverse=True
    )
    return tuple([fix_shape(res) for res in (peak, centroid, width)])
