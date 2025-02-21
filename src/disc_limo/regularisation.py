# regularisation.py
# Thomas Hilder

import numpy as np
import pylops as pl
from numpy.typing import NDArray

from .constants import π
from .dtype import FLOAT_DTYPE


def feature_weights_mat32(ω: NDArray[np.float64], s: float) -> NDArray[np.float64]:
    """
    Weights function for feature weighting of Fourier design matrix, gives Matern-3/2
    kernel in limit of infinite features.
    """
    return np.asarray(1.0 / (s**2 * ω**2 + 1.0), FLOAT_DTYPE)


def Λ_operator(
    ω: NDArray[np.float64], s: float, λ: float
) -> tuple[pl.LinearOperator, pl.LinearOperator]:
    # Λ is a diagonal matrix set by the feature weights
    Λ_diag = λ / feature_weights_mat32(ω, s * 0.5 * π) ** 2
    # Represent Λ and its inverse with Diagonal operators
    Λ = pl.Diagonal(Λ_diag)
    Λ_inv = pl.Diagonal(1 / Λ_diag)
    return Λ, Λ_inv
