# linear_solve.py
# Thomas Hilder

from timeit import default_timer as timer
from typing import Optional

import numpy as np
import pylops as pl
from numpy.typing import NDArray
from scipy.sparse.linalg import cg

from .preconditioner import Λ_operator
from .setup import Setup


def solve_linops(
    A: pl.LinearOperator,
    Y: NDArray[np.float64],
    M: Optional[pl.LinearOperator] = None,
) -> tuple[NDArray, int, float]:
    t_solve = -timer()
    X, info = cg(A, Y, M=M)
    t_solve += timer()
    return X, info, t_solve


def solve_normal_eqns(
    setup: Setup,
    Y: NDArray[np.float64],
    R: Optional[pl.Restriction] = None,
) -> tuple[NDArray, int, float]:
    """
    Solves linear system of the form
        (A.H @ Cinv @ A + Λ) X = A.H @ Cinv @ Y
    for X. We don't use a preconditioner for this case.
    """
    # Apply restriction if needed
    if R is not None:
        A = R @ setup.A
        Cinv = R @ setup.Cinv @ R.H
        Y = R @ Y
    else:
        A = setup.A
        Cinv = setup.Cinv
    # Regularisation
    Λ, _ = Λ_operator(setup.ω, setup.s, setup.λ)
    # Operator and vector for system
    op = A.H @ Cinv @ A + Λ
    vec = A.H @ Cinv @ Y
    # Solve and return solution and meta data
    return solve_linops(op, vec)


def solve_woodbury(
    setup: Setup,
    Y: NDArray[np.float64],
    M: Optional[pl.LinearOperator] = None,
    R: Optional[pl.Restriction] = None,
) -> tuple[NDArray, int, float]:
    """
    Solves linear system of the form
        (A @ Λinv @ A.H + C) α = Y
    for α, and returns X = Λinv @ A.H @ α. Preconditioner M is not required but it is
    *highly* recommended.
    """
    # Apply restriction if needed
    if R is not None:
        A = R @ setup.A
        C = R @ setup.C @ R.H
        Y = R @ Y
        M = R @ M @ R.H
    else:
        A = setup.A
        C = setup.C
    # Regularisation
    _, Λinv = Λ_operator(setup.ω, setup.s, setup.λ)
    # Operator for system
    op = A @ Λinv @ A.H + C
    # Get solution α and return X plus meta data
    α, info, t = solve_linops(op, Y, M)
    return Λinv @ A.H @ α, info, t


def linear_solve(
    setup: Setup,
    Y: NDArray[np.float64],
    M: Optional[pl.LinearOperator] = None,
    R: Optional[pl.Restriction] = None,
) -> tuple[NDArray, int, float]:
    # We decide whether to call solve_normal_eqns or solve_woodbury by whether or not we
    # have Cinv. This is not the usual advice regarding p > n or p < n TODO: elaborate
    if setup.Cinv is None:
        return solve_woodbury(setup, Y, M=M, R=R)
    else:
        return solve_normal_eqns(setup, Y, R=R)
