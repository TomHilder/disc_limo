# cube_from_weights
# Thomas Hilder

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


def convert_weights_to_channels(
    weights: NDArray[np.float64],
    design_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    TODO: Docstring! This function is user-accessible!
    """
    # Get dimension information
    n_pix_total, n_modes = design_matrix.shape
    n_channels = np.prod(weights.shape[:-1])
    n_pix_side = int(np.sqrt(n_pix_total))
    # Reshape weights to 2D of shape (n_channels, n_modes)
    weights_flat = weights.reshape((n_channels, n_modes))
    # Calculate image values for all provided channel weights
    channels = np.zeros((n_channels, n_pix_total))
    print(f"Converting {n_channels} weight vectors to images:")
    for i in tqdm(range(n_channels)):
        channels[i, :] = design_matrix @ weights_flat[i, :]
    return channels.reshape((*weights.shape[:-1], n_pix_side, n_pix_side))
