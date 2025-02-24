# results_io.py
# Thomas Hilder

import json
import pickle

import numpy as np
from astropy.io.fits.header import Header
from numpy.typing import NDArray

from .setup import Setup, setup_fit

RESULTS_EXT = "_results.npy"
SETUP_EXT = "_setup.json"
META_EXT = "_meta.json"
FITSHEADER_EXT = "_fitsheader.pkl"


def save_setup(setup: Setup, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(setup.savedata, f)


def load_setup(filename: str) -> Setup:
    with open(filename, "r") as f:
        savedata = json.load(f)
        return setup_fit(**savedata)


def save_fitsheader(header: Header, filename: str) -> None:
    with open(filename, "wb") as f:
        pickle.dump(header, f)


def load_fitsheader(filename: str) -> Header:
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_meta(meta: dict, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(meta, f)


def load_meta(filename: str) -> dict:
    with open(filename, "r") as f:
        return dict(json.load(f))


def save_results(results: NDArray, filename: str) -> None:
    np.save(filename, results)


def load_results(filename: str) -> NDArray:
    return np.asarray(np.load(filename))


def save_all(
    filename_base: str,
    results: NDArray,
    setup: Setup,
    meta: dict,
    fitsheader: Header,
) -> None:
    save_results(results, filename_base + RESULTS_EXT)
    save_setup(setup, filename_base + SETUP_EXT)
    save_meta(meta, filename_base + META_EXT)
    save_fitsheader(fitsheader, filename_base + FITSHEADER_EXT)


def load_fit(filename_base: str) -> tuple[NDArray, Setup, dict, Header]:
    return (
        load_results(filename_base + RESULTS_EXT),
        load_setup(filename_base + SETUP_EXT),
        load_meta(filename_base + META_EXT),
        load_fitsheader(filename_base + FITSHEADER_EXT),
    )
