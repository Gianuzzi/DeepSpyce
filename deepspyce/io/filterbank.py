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

import numpy as np
from deepspyce.utils.files_utils import open_file, write_to_file
from deepspyce.utils.misc import (bytes_to_data, bytes_to_header,
                                  data_to_bytes, header_to_bytes)

from .header import FILT_HEADER_TYPES

# ============================================================================
# FUNCTIONS
# ============================================================================


def _read_filterbank_header(
    path: os.PathLike, key_fmts: dict = dict(), swap: bool = False
) -> dict:
    """
    Filterbank header reader.

    Reads a filterbank header and constructs a deepheader object
    from it.

    Parameters
    ----------
    path : str or file like
        Path to the filterbank file containing the header.
    key_fmts : dict, default value = dict()
        Dictionary with key data types, to be decoded properly.
        It can also contain the string formats.
        This dictionary is used to update the standard
        filterbank header types.
    swap : bool, default value = False
        Forces swaps byte order of all readed data.

    Return
    -------
    header : ``DeepHeader class`` object.
    """
    with open_file(path, "rb") as buff:
        key_fmts = dict(FILT_HEADER_TYPES, **key_fmts)

        return bytes_to_header(buff, key_fmts, swap)


def _read_filterbank_data(
    path: os.PathLike,
    len_header: int = None,
    n_channels: int = 1,
    fmt: np.dtype = ">i8",
    order: str = "F",
    swap: bool = False,
    key_fmts: dict = dict(),
) -> np.ndarray:
    """
    Filterbank data reader.

    Reads a filterbank file and constructs a deepdata object
    from its data.

    Parameters
    ----------
    path : str or file like
        Path to the filterbank file containing the data.
    len_header : int, default value = None
        Length of the header (in bytes) to be skipped.
        If None, there is an attempt to read the header, in
        order to skip it.
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
        Dictionary with key data types, to be decoded properly.
        It can also contain the string formats.
        This dictionary is used to update the standard
        filterbank header types.
        Unused if len_header is not None.

    Return
    ------
    deepframe : ``DeepFrame class`` object.
    """
    with open_file(path, "rb") as buff:
        if len_header is None:
            key_fmts = dict(FILT_HEADER_TYPES, **key_fmts)
            bytes_to_header(buff, key_fmts, swap)
        else:
            buff.seek(len_header)
        data = bytes_to_data(buff, n_channels, fmt, order, swap)

    return data


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
    if which == "header":
        return _read_filterbank_header(path, key_fmts, swap)
    elif (which == "data") or isinstance(which, int):
        len_header = which if isinstance(which, int) else None
        return _read_filterbank_data(
            path, len_header, n_channels, fmt, order, swap, key_fmts
        )
    key_fmts = dict(FILT_HEADER_TYPES, **key_fmts)
    with open_file(path, "rb") as buff:
        header = bytes_to_header(buff, key_fmts, swap)
        data = bytes_to_data(buff, n_channels, fmt, order, swap)

    return (header, data)


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
    header_bytes = header_to_bytes(header, key_fmts, swap)
    data_bytes = data_to_bytes(data, fmt, order, swap)
    all_bytes = header_bytes + data_bytes

    write_to_file(all_bytes, outfile, "wb", overwrite)

    return
