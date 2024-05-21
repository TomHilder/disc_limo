# cube_io.py
# Thomas Hilder

from typing import Optional

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.io.fits.header import Header
from numpy.typing import NDArray

# Relationship between beam FWHM and std.
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2))
FWHM_TO_SIGMA = 1.0 / SIGMA_TO_FWHM

# Relationship between degrees and arcseconds.
DEG_TO_ARCSEC = 3600

# Number of channels to calculate RMS.
DEFAULT_NCHANNELS_NOISE = 5


def read_cube(
    filename: str,
    n_pix: Optional[int] = None,
) -> tuple[NDArray[np.float64], Header, Gaussian2DKernel, float, int, int, int]:
    # Open cube and read image and header
    cube = fits.open(filename)
    image, header = cube[0].data, cube[0].header
    # Get some needed quantities
    n_x, n_y, n_channels = read_nspaxels(header)
    beam = Gaussian2DKernel(*read_beam(header, 1))
    rms = estimate_rms(image)
    # Check image is square (we need this)
    assert n_x == n_y
    if n_pix is not None:
        # Trim cube
        centre_index = n_x // 2
        lower_index = centre_index - (n_pix // 2)
        upper_index = centre_index + (n_pix // 2)
        image = image[:, lower_index:upper_index, lower_index:upper_index]
        n_x, n_y = image[0, :, :].shape
    # Return everything
    return image, header, beam, rms, n_x, n_y, n_channels


def upsampled_beam(header: Header, n_pix: int, n_eval: int) -> Gaussian2DKernel:
    upscale_factor = float(n_eval / n_pix)
    return Gaussian2DKernel(*read_beam(header, upscale_factor))


def read_pixelscale(cube_header: Header) -> float:
    """Read the pixelscale of the cube."""
    return float(cube_header["CDELT2"]) * DEG_TO_ARCSEC


def read_beam(cube_header: Header, scale: float = 1) -> tuple[float, float, float]:
    """
    Read the beam from cube header. Returns (bmaj, bmin, bpa) with bmaj and bmin as
    standard deviations in pixels, and bpa in radians.
    """
    pixelscale = read_pixelscale(cube_header)
    return (
        scale * cube_header["BMAJ"] * DEG_TO_ARCSEC * FWHM_TO_SIGMA / pixelscale,
        scale * cube_header["BMIN"] * DEG_TO_ARCSEC * FWHM_TO_SIGMA / pixelscale,
        np.deg2rad(cube_header["BPA"]),
    )


def read_nspaxels(cube_header: Header) -> tuple[int, int, int]:
    """
    Read the number of spaxels in the image from the header. Returns
    (n_x, n_y, n_channels).
    """
    return (
        int(cube_header["NAXIS1"]),
        int(cube_header["NAXIS2"]),
        int(cube_header["NAXIS3"]),
    )


def estimate_rms(
    image: NDArray[np.float64], nchannels: int = DEFAULT_NCHANNELS_NOISE
) -> float:
    """
    Estimate the RMS noise of the cube from the first and last 5 channels by default.
    """
    # Estimate RMS using standard deviation of channels far from systemic velocity
    return float(
        np.nanstd(
            a=np.concatenate(
                [
                    image[:nchannels, :, :],
                    image[-nchannels:, :, :],
                ]
            )
        )
    )
