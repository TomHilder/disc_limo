# constants.py
# Thomas Hilder

import multiprocessing

import numpy as np

# Math
π = np.pi

# Numerical tolerances
FINUFFT_TOL = 1e-10

# Relationship between beam FWHM and std
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2))
FWHM_TO_SIGMA = 1.0 / SIGMA_TO_FWHM

# Relationship between degrees and arcseconds
DEG_TO_ARCSEC = 3600

# Number of threads for parallelism
N_THREADS = multiprocessing.cpu_count()
# N_THREADS = 1
