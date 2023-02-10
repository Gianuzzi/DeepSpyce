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

"""Module with filterbank (.fil) I/O functions."""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import warnings

from deepspyce.utils.files_utils import open_file, write_to_file
from deepspyce.utils.misc import (
    bytes_to_data,
    bytes_to_dict,
    data_to_bytes,
    dict_to_bytes,
)

import numpy as np

from .header import FILT_HEADER_TYPES

# ============================================================================
# FUNCTIONS
# ============================================================================


def read_filterbank(
    path: os.PathLike,
    n_channels: int = 2048,
    fmt: np.dtype = ">i8",
    order: str = "F",
    swap: bool = False,
    key_fmts: dict = dict(),
    which: str = "both",
) -> tuple:
    """
    Filterbank reader.

    Reads a filterbank file and constructs a deepframe object
    from its data and its header.

    Parameters
    ----------
    path : str or file like
        Path to the filterbank file containing the data.
    n_channels : int, default value = 2048
        Number of channels (cols) of the data.
    fmt : format, default value = ">i8"
        Data type of the value to be decoded.
        It can also be a string format.
    order : {"C", "F"}, default value = "F"
        Read the data elements using this index order.
        "C" indicates C-like index order.
        "F" indicates Fortran-like index order.
    swap : bool, default value = False
        Indicates if the byteorder of the data read is returned swapped.
        It can also be a string of the wanted byte order.
    key_fmts : dict, default value = dict()
        Dictionary with extra key data types, to be decoded properly.
        It can also contain the string formats.
        This dictionary is used to update the standard
        filterbank header types.
    which : str, default value = "both"
        Whether to read only the "header", the "data" or "both".
        It can be an int equal to the header lenght to be skipped.

    Return
    ------
    deepframe : ``DeepFrame class`` object.
    """
    key_fmts = dict(FILT_HEADER_TYPES, **key_fmts)

    with open_file(path, "rb") as buff:
        if isinstance(which, int):
            buff.seek(which)
        else:
            header = bytes_to_dict(buff, key_fmts, swap)
        data = bytes_to_data(buff, n_channels, fmt, order, swap)

    if which == "header":
        return header
    if (which == "data") or isinstance(which, int):
        return data
    if which == "both":
        return (header, data)

    return


def write_filterbank(
    data: np.ndarray,
    header: dict = dict(),
    key_fmts: dict = dict(),
    outfile: os.PathLike = None,
    fmt: np.dtype = ">i8",
    order: str = "F",
    swap: bool = False,
    overwrite: bool = False,
):
    """
    Filterbank file writer.

    Writes a numpy ndarray data and
    a header dict into a filterbank file.

    Parameters
    ----------
    data : numpy ndarray
        Array with data to be written into filterbank file.
    header : dict
        Dictionary of header to be stored into filterbank file.
    key_fmts : dict, default value = dict()
        Dictionary with extra key data types, to be decoded properly.
        It can also contain the string formats.
        This dictionary is used to update the standard
        filterbank header types.
    outfile : str or file like
        Path or file like objet to the filterbank to be written.
    fmt : format, default value = ">i8"
        Data type format. If None, it is infered from the data.
    order : {"C", "F"}, default value = "F"
        Read the data elements using this index order.
        "C" indicates C-like index order.
        "F" indicates Fortran-like index order.
    swap : bool, default value = False
        Indicates if the byteorder of the data to be written is swapped.
        It can also be a string of the wanted byte order.
    overwrite : bool, default value = False
        Indicates if, in case the outfile already exists, it is overwriten.
    """
    if not isinstance(header, dict):
        raise TypeError(
            "Header must be a dict. Type {} was given".format(type(header))
        )
    name = header.get("rawdatafile", None)
    if outfile is None:
        if name is None:
            raise OSError("Could not resolve/infer output file name.")
        else:
            outfile = name
            filen = outfile
    elif hasattr(outfile, "name"):
        filen = os.path.basename(getattr(outfile, "name"))
    else:
        filen = os.path.basename(outfile)
    if filen != name:
        warnings.warn(
            f"\nFile name: '{filen}' and "
            + f"\nrawdatafile: '{name}' in header "
            + "\ndo not match."
        )
    key_fmts = dict(FILT_HEADER_TYPES, **key_fmts)
    header_bytes = dict_to_bytes(header, key_fmts, swap)
    data_bytes = data_to_bytes(data, fmt, order, swap)
    all_bytes = header_bytes + data_bytes

    write_to_file(all_bytes, outfile, "wb", overwrite)

    return
