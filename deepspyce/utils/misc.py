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

"""Module with misc functions."""

# =============================================================================
# IMPORTS
# =============================================================================

import io
import json
import os
import pickle
import struct
import warnings

import numpy as np

from .files_utils import is_readable, read_file, write_to_file

# ============================================================================
# UTILS
# ============================================================================


def _my_fmt(dtype: np.dtype, order: str = "|"):
    """
    Simple dtype -> fmt converter.

    Converts a given type to string format (fmt)

    Parameters
    ----------
    dtype : type
        Type to convert. It can also be the string format (fmt) istelf.
    neworder : str, default value = "|"
        Byte order to force, a value from the byte order specifications
        below:
        * 'S' - swap dtype from current to opposite endian
        * {'<', 'little'} - little endian
        * {'>', 'big'} - big endian
        * '=' - native order
        * {'|', 'I'} - ignore (no change to byte order)

    Return
    ------
    fmt : str
        String format of the given type.
    """
    dtype = np.dtype(dtype)
    char = dtype.char
    if order in [None, "s", True]:
        order = "S"
    if order:
        dtype = dtype.newbyteorder(order)
    fmt = dtype.byteorder + char

    return fmt


def _my_pack(value: any, fmt: np.dtype = None, order: str = "|") -> bytes:
    """
    Bytes packer.

    Packes value according to fmt. Useful for filterbank encoding.

    Parameters
    ----------
    value : any
        Value to be packed.
        If a number, it is only packed.
        If a string, then its length is packed and its value encoded.
    fmt : numpy dtype, default value = None
        Data type of the value to be decoded.
        It can also be a string format.
        If None, there is an attemp to infer it from the value.
    order : str, default value = "|"
        Byte order to force.

    Return
    -------
    packed : bytes
        Packed data.
    """
    if fmt is None:
        fmt = type(value)
    fmt = _my_fmt(fmt, order)
    if "U" in fmt:
        return struct.pack(fmt[0] + "I", len(value)) + value.encode()

    return struct.pack(fmt, value)


def _my_decode(encoded: bytes, fmt: np.dtype = "U", order: str = "|") -> any:
    """
    Bytes decoder (from reading).

    Reads the first (next) encoded value, from bytes.

    Parameters
    ----------
    encoded : bytes
        Bytes of header to be decoded.
        It can also be a readable io.BytesIO (or Buffered) type. If so,
        the file pointer is moved while reading.
    fmt : numpy dtype, default value = "U"
        Data type of the value to be decoded.
        It can also be a string format.
    order : str, default value = "|"
        Byte order to force.

    Return
    -------
    value : any
        Decoded value.
    """
    if isinstance(encoded, bytes):
        encoded = io.BytesIO(encoded)
    fmt = _my_fmt(fmt, order)
    if "U" in fmt:
        val_nchar = struct.unpack(fmt[0] + "I", encoded.read(4))[0]
        val = encoded.read(val_nchar).decode()
    else:
        val_nchar = struct.calcsize(fmt)
        val = struct.unpack(fmt, encoded.read(val_nchar))[0]

    return val


def _swap(value, fmt: np.dtype = None) -> any:
    """
    Bytes swapper.

    Swaps the bytes of a value and return the new value.

    Parameters
    ----------
    value : any
        Value to be swaped.
    fmt : numpy dtype, default value = None
        Data type of the value to be decoded.
        It can also be a string format.
        If None, there is an attemp to infer it from the value.

    Return
    -------
    swapped : any
        New value, after swapping bytes.
    """
    if fmt is None:
        fmt = type(value)
    fmt = _my_fmt(fmt)
    fmt2 = _my_fmt(fmt, "S")

    return struct.unpack(fmt2, struct.pack(fmt, value))[0]


def _all_keys_str(dicc: dict) -> bool:
    """
    Checks if all keys of a given dictionary are strings.
    """
    return all([isinstance(key, str) for key in dicc.keys()])


def _check_key_pos(dicc: dict, key: str, pos: int, verb: bool = False) -> bool:
    """
    Dictionary key position check.

    Checks if a given key is at a specific position of a dict's keys index.

    Parameters
    ----------
    dicc : dict
        Dictionary to be checked.
    key : str
        Dictionary's key to be checked.
    pos : int
        Dictionary key's position to be checked.
    verb : bool, default value = False
        Indicates if a warning is raised.

    Return
    -------
    result : bool
        Bool indicating if the given key is in the specified location.
        None if the key is missing.
    """
    keys = list(dicc.keys())
    loc = keys.index(key) if key in keys else None
    if loc == pos:
        return True
    if loc is None:
        if verb:
            warnings.warn(f"{key} is missing in the dictionary.")
        return None
    if verb:
        warnings.warn(f"{key} is not in the {pos} dictionary key.")

    return False


# ============================================================================
# FUNCTIONS
# ============================================================================


def check_dict_types(header: dict, key_fmts: dict, warn: bool = True) -> bool:
    """
    Dict data types checker.

    Checks that the main entries of a given dictionary
    have their correct data type, according to a dict of types.
    Raises a warning for each incorrect data type.

    Parameters
    ----------
    header : dict
        Dictionary to be checked.
    key_fmts : dict, default value = FILT_HEADER_TYPES
        Dictionary with key types as values.
    warn : bool, default value = True
        Raise a warning for any value type not matching.

    Return
    ------
    good : bool
        Boolean indicating if everything is ok.
    """
    for key, value in header.items():
        if key in key_fmts.keys():
            dtype = key_fmts.get(key)
            if (value is not None) and not isinstance(value, dtype):
                if warn:
                    warnings.warn(
                        "WARNING. Key %s value should be type %s"
                        % (key, dtype)
                    )
                good = False
        # elif (key in ["HEADER_START", "HEADER_END"]) and (value is not None):
        #     warnings.warn("WARNING. %s key value should be None" %key)
        #     good = False
        else:
            if warn:
                warnings.warn("WARNING. Unexpected key: %s" % key)
            good = False

    return good


def dict_to_bytes(
    header: dict, key_fmts: dict = dict(), swap: bool = False
) -> bytes:
    """
    Dictionary (header) encoder.

    Encodes a dictionary, as a filterbank header.

    Parameters
    ----------
    header : dict
        Dictionary to be encoded.
    key_fmts : dict, default value = dict()
        Dictionary with key  data types, to be encoded properly.
        If not given (or a key is missing or its value is None),
        then there is an attemp to infer it from the value (of the header).
        It can also contain the string formats.
    swap : bool, default value = False
        Forces swaps byte order of all writing data.

    Return
    -------
    encoded : bytes
        Dictionary encoded into bytes (filterbank header like).
    """
    kvheader = dict({k: v for k, v in header.items() if v is not None})
    swap = "S" if swap else "|"
    encoded = _my_pack("HEADER_START", str, swap)
    for key, value in kvheader.items():
        if key in ["HEADER_START", "HEADER_END"]:
            continue
        fmt = key_fmts.get(key, None)
        encoded += _my_pack(str(key), str, swap) + _my_pack(value, fmt, swap)
    encoded += _my_pack("HEADER_END", str, swap)

    return encoded


def bytes_to_dict(
    encoded: bytes, key_fmts: dict = dict(), swap: bool = False
) -> dict:
    """
    Header (dictionary) bytes decoder.

    Decodes a dictionary, as a filterbank header.

    Parameters
    ----------
    encoded : bytes
        Bytes of header to be decoded.
        It can also be a readable io.BytesIO (or Buffered) type. If so,
        the file pointer is moved while reading.
    key_fmts : dict, default value = dict()
        Dictionary with key data types, to be decoded properly.
        If not given (or a key is missing), then float64 type is assumed.
        It can also contain the string formats.
    swap : bool, default value = False
        Forces swaps byte order of all readed data.

    Return
    -------
    header : dict
        Dictionary of the decoded header.
    """
    header = dict()
    if isinstance(encoded, bytes):
        encoded = io.BytesIO(encoded)
    swap = "S" if swap else "|"
    try:
        key0 = _my_decode(encoded, str, swap)
    except UnicodeDecodeError as e:
        print("Error:" + str(e))
        warnings.warn("WARNING. Trying swapping bytes order...")
        swap = "|" if swap == "S" else "S"
        encoded.seek(0)
        key0 = _my_decode(encoded, str, swap)
    if key0 != "HEADER_START":
        warnings.warn("WARNING. Header does not start with 'HEADER_START'")
        bad_header = True
        key0_type = key_fmts.get(key0, float)
        val0 = _my_decode(encoded, key0_type, swap)
        header[key0] = val0
    else:
        bad_header = False
    while True:
        if bad_header:
            pos = encoded.tell()
            if encoded.read(1) == b"":
                break
            encoded.seek(pos)
        key = _my_decode(encoded, str, swap)
        if key == "HEADER_END":
            break
        val = _my_decode(encoded, key_fmts.get(key, float), swap)
        header[key] = val

    return header


def data_to_bytes(
    data: np.ndarray,
    fmt: np.dtype = None,
    order: str = "F",
    swap: bool = False,
) -> bytes:
    """
    Array to bytes converter.

    Converts common numpy ndarray to bytes, using ``np.asarray``.

    Parameters
    ----------
    data : array_like
        Input data, in any form that can be converted to an array.
        This includes lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists and ndarrays.
    fmt : data-type, defalut value = None
        Data type of the values to be encoded.
        It can also be a string format.
        If None, there is an attemp to infer it from the values.
    order : {"C", "F"}, default value = "F"
        Write the data elements using this index order:
        "C" indicates C-like index order.
        "F" indicates Fortran-like index order.
    swap : bool, default value = False
        Indicates if the byteorder of the data read is returned swapped.
        It can also be a string of the wanted byte order.

    Return
    ------
    databytes : bytes
        Bytes of the converted data.
    """
    if swap:
        fmt = np.dtype(fmt).newbyteorder("S" if swap is True else swap)

    return np.asarray(data, dtype=fmt).tobytes(order)


def bytes_to_data(
    encoded: bytes,
    n_cols: int = 1,
    fmt: np.dtype = ">i8",
    order: str = "F",
    swap: bool = False,
) -> np.ndarray:
    """
    Bytes to numpy ndarray converter.

    Decodes an encoded array.

    Parameters
    ----------
    encoded : bytes
        Bytes of header to be decoded.
        It can also be a readable io.BytesIO (or Buffered) type. If so,
        the file pointer is moved while reading.
    n_cols : int, default value = 1
        Number of columns of the data.
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

    Return
    ------
    array : numpy ndarray
        Numpy ndarray of the decoded bytes.
    """
    if is_readable(encoded):
        encoded = encoded.read()
    dt = np.dtype(fmt)
    if swap:
        dt = dt.newbyteorder("S" if swap is True else swap)
    data = np.frombuffer(encoded, dtype=dt)
    bytes_per_data = dt.alignment
    total_bytes = len(encoded)
    n_records = int(total_bytes / n_cols / bytes_per_data)
    array = data.reshape(n_cols, n_records, order=order)
    if n_cols == 1:
        return array.flatten()

    return array


def dict_to_file(
    dicc: dict,
    path_or_stream: os.PathLike,
    sep: str = ",",
    ext: str = "csv",
    overwrite: bool = False,
    **kwargs,
):
    """
    Write a dictionary into a file.

    Parameters
    ----------
    dicc : dict
        Dictionary to be written into file.
    path_or_stream : str or file like
        Path to the file.
    sep : str
        Field delimiter for the output file.
        Unused if format is not "csv".
    ext : {"csv", "json", "pickle", "filterbank"}, default value = "csv"
        Outfile format.
    overwrite : bool, default value = False
        Indicates if, in case the outfile already exists, it is overwriten.
    kwargs : dict (optional)
        Parameters to send to the subjacent ``bytes_to_dict()`` function.
        Unused if ext is not "filterbank".
    """
    mode = "w"
    if ext == "json":
        dump = json.dumps(dicc)
    elif ext == "txt":
        dump = str(dicc)
    elif ext == "pickle":
        dump = pickle.dumps(dicc)
        mode = "wb"
    elif ext == "filterbank":
        dump = dict_to_bytes(dicc, **kwargs)
        mode = "wb"
    elif ext == "csv":
        dump = ""
        for key, value in dicc.items():
            dump += "%s%s%s\n" % (key, sep, value)
    else:
        raise ValueError("Format %s not supported." % ext)

    write_to_file(dump, path_or_stream, mode, overwrite)

    return


def dict_from_file(
    path_or_stream: os.PathLike,
    sep: str = ",",
    ext: str = "csv",
    renum: bool = True,
    **kwargs,
) -> dict:
    """
    Read a dictionary from a file.

    Parameters
    ----------
    path_or_stream : str or file like
        Path to the file.
    sep : str
        Field delimiter for input file.
        Unused if ext is not "csv".
    ext : {"csv", "json", "pickle", "filterbank"}, default value = "csv"
        Outfile format.
    renum : bool, default value = True
        Converts all possible numerical values into floats.
    kwargs : dict (optional)
        Parameters to send to the subjacent ``bytes_to_dict()`` function.
        Unused if ext is not "filterbank".

    Return
    ------
    dicc : dict
        Dictionary of the file readed.
    """
    mode = "rb" if ext in ["pickle", "filterbank"] else "r"
    data = read_file(path_or_stream, mode)
    if ext == "json":
        dicc = json.loads(data)
    elif ext == "pickle":
        dicc = pickle.loads(data)
    elif ext == "filterbank":
        dicc = bytes_to_dict(data, **kwargs)
    elif ext == "csv":
        dicc = dict([line.split(sep) for line in data.splitlines()])
    else:
        raise ValueError("Format %s not supported." % ext)

    if renum:
        for key, value in dicc.items():
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
                dicc[key] = value
            except ValueError:
                continue

    return dicc
