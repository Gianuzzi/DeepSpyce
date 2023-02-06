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

"""Module with IAR (.iar) I/O functions."""

# =============================================================================
# IMPORTS
# =============================================================================

import os

from deepspyce.utils.misc import dict_from_file, dict_to_file

# ============================================================================
# UTILS
# ============================================================================

_header_types = dict(
    {
        "Source Name": str,
        "Source RA (hhmmss.s)": float,
        "Source DEC (ddmmss.s)": float,
        "Reference DM": float,
        "Pulsar Period": float,
        "Highest Observation Frequency (MHz)": float,
        "Telescope ID": int,
        "Machine ID": int,
        "Data Type": int,
        "Observing Time (minutes)": float,
        "Local Oscillator (MHz)": float,
        "Gain (dB)": float,
        "Total Bandwith (MHz)": float,
        "Average Data": int,
        "Sub Bands": int,
        "Cal": int,
    }
)

# ============================================================================
# FUNCTIONS
# ============================================================================


def read_iar(path: os.PathLike, **kwargs) -> dict:
    """
    Iar file reader.

    Reads a .iar file and builds a deepheader from its data.

    Parameters
    ----------
    path : str or file like
        Path to the .iar file.
    kwargs :
        Extra arguments to the function
        ``deepspyce.utils.misc.dict_from_file()``

    Return
    -------
    header : ``DeepHeader class`` object.
    """

    return dict_from_file(path, **kwargs)


def write_iar(dicc: dict, outfile: os.PathLike, **kwargs):
    """
    Iar file writer.

    Writes a .iar file, from a dictionary.

    Parameters
    ----------
    dicc : dict
        Dictionary with IAR file data to be written.
    outfile : str or file like
        Path or file like objet to the iar file to store the iar header.
    kwargs :
        Extra arguments to the function
        ``deepspyce.utils.misc.dict_to_file()``
    """
    dict_to_file(dicc, outfile, **kwargs)

    return
