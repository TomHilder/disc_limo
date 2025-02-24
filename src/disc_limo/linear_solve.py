# linear_solve.py
# Thomas Hilder

from timeit import default_timer as timer

import numpy as np
import pylops as pl
from numpy.typing import NDArray
from scipy.sparse.linalg import cg

from .preconditioner import Λ_operator
from .setup import Setup


def solve_linops(
    A: pl.LinearOperator,
    Y: NDArray[np.float64],
    M: pl.LinearOperator = None,
) -> tuple[NDArray, int, float]:
    t_solve = -timer()
    X, info = cg(A, Y, M=M)
    t_solve += timer()
    return X, info, t_solve


def solve_normal_eqns(
    setup: Setup,
    Y: NDArray[np.float64],
) -> tuple[NDArray, int, float]:
    """
    Solves linear system of the form
        (A.H @ Cinv @ A + Λ) X = A.H @ Cinv @ Y
    for X. We don't use a preconditioner for this case.
    """
    # Regularisation
    Λ, _ = Λ_operator(setup.ω, setup.s, setup.λ)
    # Operator and vector for system
    op = setup.A.H @ setup.Cinv @ setup.A + Λ
    vec = setup.A.H @ setup.Cinv @ Y
    # Solve and return solution and meta data
    return solve_linops(op, vec)


def solve_woodbury(
    setup: Setup,
    Y: NDArray[np.float64],
    M: pl.LinearOperator = None,
) -> tuple[NDArray, int, float]:
    """
    Solves linear system of the form
        (A @ Λinv @ A.H + C) α = Y
    for α, and returns X = Λinv @ A.H @ α. Preconditioner M is not required but it is
    *highly* recommended.
    """
    # Regularisation
    _, Λinv = Λ_operator(setup.ω, setup.s, setup.λ)
    # Operator for system
    op = setup.A @ Λinv @ setup.A.H + setup.C
    # Get solution α and return X plus meta data
    α, info, t = solve_linops(op, Y, M)
    return Λinv @ setup.A.H @ α, info, t


def linear_solve(
    setup: Setup,
    Y: NDArray[np.float64],
    M: pl.LinearOperator = None,
) -> tuple[NDArray, int, float]:
    # We decide whether to call solve_normal_eqns or solve_woodbury by whether or not we
    # have Cinv. This is not the usual advice regarding p > n or p < n TODO: elaborate
    if setup.Cinv is None:
        return solve_woodbury(setup, Y, M)
    else:
        return solve_normal_eqns(setup, Y)
