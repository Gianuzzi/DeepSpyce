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

"""Module with file managing functions."""

# =============================================================================
# IMPORTS
# =============================================================================

import io
import os

# ============================================================================
# CONSTANTS
# ============================================================================

rmodes = ["r", "rb"]

wmodes = ["w", "wb", "w+", "wb+", "r+"]

amodes = ["a", "ab", "a+", "ab+"]

# ============================================================================
# FUNCTIONS
# ============================================================================


def is_filelike(fileobj: io.IOBase) -> bool:
    """
    Check if input is a filelike object.

    Parameters
    ----------
    fileobj : file like
        File object.
    """

    return issubclass(type(fileobj), io.IOBase)


def is_opened(fileobj: io.IOBase) -> bool:
    """
    Check if a filelike object is opened.

    Parameters
    ----------
    fileobj : file like
        File object.
    """

    return not getattr(fileobj, "closed", True)


def is_writable(fileobj: io.FileIO) -> bool:
    """
    Check if a filelike object is writable.

    Parameters
    ----------
    fileobj : file like
        File object.
    """

    return getattr(fileobj, "writable", False)


def is_readable(fileobj: io.FileIO) -> bool:
    """
    Check if a filelike object is readable.

    Parameters
    ----------
    fileobj : file like
        File object.
    """

    return getattr(fileobj, "readable", False)


def file_exists(path_str: os.PathLike) -> bool:
    """
    Check if a file exists.

    Parameters
    ----------
    path_or_stream : str or file like
        Path to the file.
    """

    return os.path.isfile(path_str)


def open_file(
    path_str: os.PathLike, mode: str = "r", overwrite: bool = False
) -> io.IOBase:
    """
    Open a file.

    Parameters
    ----------
    path_or_stream : str or file like
        Path to the file.
    mode : str, default value = "r"
        Python opening mode.
    overwrite : bool, default value = False
        Indicates if, in case the outfile already exists, it is overwriten.

    Return
    ------
    opened : io.IOBase
        Opened file.
    """
    if (mode in amodes + wmodes) and (not overwrite) and file_exists(path_str):
        raise FileExistsError("File will not be overwritten.")

    if is_filelike(path_str):
        if not is_opened(path_str):
            return open(path_str.name, mode)
        return path_str

    return open(path_str, mode)


def read_file(path_or_stream: os.PathLike, mode: str = "r") -> any:
    """
    Read data from a file.

    Parameters
    ----------
    path_or_stream : str or file like
        Path to the file.
    mode : str, default value = "r"
        Python reading mode.

    Return
    ------
    data : str
        Readed data from file.
    """
    if is_readable(path_or_stream):
        return path_or_stream.read()
    if isinstance(path_or_stream, (str, os.PathLike)):
        with open_file(path_or_stream, mode) as buff:
            data = buff.read()
        return data

    raise OSError(f"Could not read {path_or_stream}")


def write_to_file(
    data: any,
    path_or_stream: os.PathLike,
    mode: str = "w",
    overwrite: bool = False,
):
    """
    Write data into a file.

    Parameters
    ----------
    data : any
        Data to be written.
    path_or_stream : str or file like
        Path to the file.
    mode : str, default value = "w"
        Python writing mode.
    overwrite : bool, default value = False
        Indicates if, in case the outfile already exists, it is overwriten.
    """
    if is_writable(path_or_stream):
        path_or_stream.write(data)
        return
    if isinstance(path_or_stream, (str, os.PathLike)):
        with open_file(path_or_stream, mode, overwrite) as buff:
            buff.write(data)
        return

    raise OSError(f"Could not write data into {path_or_stream}")
