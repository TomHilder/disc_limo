# preconditioner.py
# Thomas Hilder

from timeit import default_timer as timer
from typing import Callable

import numpy as np
import pylops as pl
from tqdm import tqdm

from .constants import N_THREADS
from .regularisation import Λ_operator
from .setup import Setup


def block_jacobi_preconditioner(
    op: pl.LinearOperator, block_size: int
) -> pl.LinearOperator:
    """
    Build a block-Jacobi preconditioner for a square linear operator. Preconditioner
    returned as a block diagonal linear operator.
    """
    # Check input operator is square
    if op.shape[0] != op.shape[1]:
        raise ValueError("op must be square.")
    n = op.shape[0]
    # Store the blocks in a list
    blocks = []
    # Partition the indices into blocks
    for start in tqdm(range(0, n, block_size)):
        indices = np.arange(start, min(n, start + block_size))
        m = len(indices)
        # Build the block by computing columns using the operator's matvec
        block = np.zeros((m, m))
        for j, col in enumerate(indices):
            # Column extracted via matvec on corresponding unit vector ξ
            ξ = np.zeros(n)
            ξ[col] = 1.0
            col_full = op.matvec(ξ)
            block[:, j] = col_full[indices]
        # Invert the block and store
        inv_block = np.linalg.inv(block)
        blocks.append(inv_block)
    # Construct preconditioner as a block diagonal operator
    return pl.BlockDiag(blocks, nproc=N_THREADS)


def M_operator(
    setup: Setup,
    precon: Callable = block_jacobi_preconditioner,
    **precon_kwargs,
) -> pl.LinearOperator:
    """
    Build preconditioner for solving linear system with Woodbury form. That is, we are
    solving
        (A @ Λ^-1 @ A.H + C) α = Y
    for α. Therefore we want preconditioner M where
        M ≈ (A @ Λ^-1 @ A.H + C)^-1
    This function is just a convenient wrapper that takes the Setup object instead.
    """
    # Regularisation/prior operator
    _, Λ_inv = Λ_operator(setup.ω, setup.s, setup.λ)
    # Operator for A @ Λ^-1 @ A.H + C (called the Schur complement)
    schur = setup.A @ Λ_inv @ setup.A.H + setup.C
    # Get preconditioner approximating inverse of Schur complement
    return precon(schur, **precon_kwargs)
