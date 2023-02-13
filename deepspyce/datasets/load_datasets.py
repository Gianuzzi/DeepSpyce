#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   DeepSpyce Project (https://github.com/suaraujo/DeepSpyce).
# Copyright (c) 2020, Susana Beatriz Araujo Furlan
# License: MIT
#   Full Text: https://github.com/suaraujo/DeepSpyce/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""Module with functions to load provided datasets."""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

from deepspyce.io import read_raw
from deepspyce.utils.misc import dict_from_file

import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# ============================================================================
# FUNCTIONS
# ============================================================================


def load_raw_1m() -> np.ndarray:
    """
    Template raw_1m data file loader.

    Loads the template raw_1m data file.

    Parameters
    ----------
    raw : bool, default value = False
        Indicates if the raw data is returned bytes.

    Return
    ------
    array : ``numpy.ndarray``
        Numpy ndarray of the aw_1m file data.
    """
    return read_raw(PATH / "20201027_133329_1m.raw")


def load_raw_test() -> np.ndarray:
    """
    Template raw_test data file loader.

    Loads the template raw_test data file.

    Return
    ------
    array : ``numpy.ndarray``
        Numpy ndarray of the raw_test file data.
    """
    return read_raw(PATH / "20201027_133329_test.raw")


def load_csv_test() -> np.ndarray:
    """
    Template .csv data file loader.

    Loads the template csv_test data file.

    Return
    ------
    array : ``numpy.ndarray``
        Numpy ndarray of the csv_test file data.
    """
    return np.loadtxt(
        PATH / "20201027_133329_test.csv", dtype=">i8", delimiter=","
    )


def load_iar() -> dict:
    """
    Template .iar data file loader.

    Loads the template .iar data file as a dict.

    Return
    ------
    iar : dict
        Dictionary of the template iar file.
    """
    return dict_from_file(PATH / "J0437-4715_1_A1.iar")
