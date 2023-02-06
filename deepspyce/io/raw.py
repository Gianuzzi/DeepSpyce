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

"""Module with raw (.raw) I/O functions."""

# =============================================================================
# IMPORTS
# =============================================================================

import os

from deepspyce.utils.files_utils import read_file, write_to_file
from deepspyce.utils.misc import bytes_to_data, data_to_bytes

import numpy as np

# ============================================================================
# FUNCTIONS
# ============================================================================


def read_raw(
    path: os.PathLike,
    n_channels: int = 2048,
    fmt: np.dtype = ">i8",
    order: str = "F",
    swap: bool = False,
) -> np.ndarray:
    """
    Raw file reader.

    Reads a binary raw file and constructs a deepdata object
    from its data.

    Parameters
    ----------
    path : str or file like
        Path to the raw file containing the data.
    n_channels : int, default value = 2048
        Number of channels (columns) of the data.
    fmt : format, default value = ">i8"
        Data type format.
    order : {"C", "F"}, default value = "F"
        Read the data elements using this index order.
        "C" indicates C-like index order.
        "F" indicates Fortran-like index order.
    swap : bool, default value = False
        Indicates if the byteorder of the data read is returned swapped.
        It can also be a string of the wanted byte order.

    Return
    ------
    deepdata : ``DeepData class`` object.
    """
    encoded = read_file(path, "rb")

    return bytes_to_data(encoded, n_channels, fmt, order, swap)


def write_raw(
    data: np.ndarray,
    outfile: os.PathLike,
    fmt: np.dtype = ">i8",
    order: str = "F",
    swap: bool = False,
    overwrite: bool = False,
):
    """
    Raw file writer.

    Writes a pandas DataFrame (or numpy ndarray) data
    into a .raw file.

    Parameters
    ----------
    data : pandas DataFrame or numpy ndarray
        DataFrame or ndarray to be written into raw file.
    outfile : str or file like
        Path or file like objet to the file to store the raw data.
    fmt : data-type, default value = ">i8"
        Data type format. If None, it is infered from the data.
    order : {"C", "F"}, default value = "F"
        Write the data elements using this index order.
        "C" indicates C-like index order.
        "F" indicates Fortran-like index order.
    swap : bool, default value = False
        Indicates if the byteorder of the data to be written is swapped.
        It can also be a string of the wanted byte order.
    overwrite : bool, default value = False
        Indicates if, in case the outfile already exists, it is overwriten.
    """
    encoded = data_to_bytes(data, fmt, order, swap)
    write_to_file(encoded, outfile, "wb", overwrite)

    return
