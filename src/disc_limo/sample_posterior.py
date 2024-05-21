# sample_posterior.py
# Thomas Hilder

from typing import Any, Optional

import numpy as np
from numpy.linalg import LinAlgError, cholesky, eigh
from numpy.typing import NDArray
from scipy.linalg import issymmetric
from tqdm import tqdm


def decompose_covariance_matrix(Sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Decompose the covariance matrix as Sigma = A @ A.T since we can use A to sample from
    the multivariate normal defined by Sigma easily.
    """
    # Sigma should be symmetric but may not be due to numerical error
    if not issymmetric(Sigma):
        Sigma = 0.5 * (Sigma + Sigma.T)
    # If the matrix is positive definite then Cholesky is cheapest and gives A directly
    # since Sigma is real so Sigma = A @ A.conj().T = A @ A.T
    try:
        return cholesky(Sigma)
    # If the matrix is only positive semi-definite then Cholesky will raise Exception
    except LinAlgError:
        # In this case we do a Eigendecomposition Sigma = Q @ Lambda @ Q^-1 but since
        # Sigma is real and symmetric Q^-1 = Q.T. Thus we can set A = Q @ Lambda^1/2
        eigenvalues, Q = eigh(Sigma)
        Lambda_half = Q @ np.diag(np.sqrt(eigenvalues))
        return np.asarray(Q @ Lambda_half, np.float64)


def get_posterior_samples(
    weights_vectors: NDArray[np.float64],
    weights_covariances: NDArray[np.float64],
    n_samples: int,
    seed: Optional[Any] = None,
) -> NDArray[np.float64]:
    """
    TODO: Docstring! This function is user-accessible!
    """
    rng = np.random.default_rng(seed)
    # Get Covariance matrix decomposition since it is the same for all channels
    L = decompose_covariance_matrix(weights_covariances)
    # Iterate to create sample cubes, could probably be parallelised but it's pretty
    # fast anyway
    samples = []
    print(f"Generating {n_samples} posterior samples of Fourier weights:")
    for _ in tqdm(range(n_samples)):
        # Vectors full of standard normal samples
        alpha_vectors = rng.standard_normal(size=weights_vectors.shape).T
        # Transform to desired multivariate normal using covariance matrix decomposition
        samples.append((weights_vectors.T + L @ alpha_vectors).T)
    return np.array(samples)
