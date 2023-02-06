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

"""Module with fits (.fits) I/O functions."""

# =============================================================================
# IMPORTS
# =============================================================================

import os
from datetime import datetime

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


# ============================================================================
# UTILS
# ============================================================================

_header_template = dict(
    {
        "SIMPLE": ("T", "/ conforms to FITS standard"),
        "BITPIX": (8, "/ BITS/PIXEL"),
        "NAXIS": (0, "/ number of array dimensions"),
        "EXTEND": ("T", "/File contains extensions"),
        "DATE": (datetime.today().strftime("%y-%m-%d"), "/"),
        "ORIGIN": ("IAR", "/ origin of observation"),
        "TELESCOP": ("Antena del IAR", "/ the telescope used"),
        "OBSERVAT": ("IAR", "/ the observatory"),
        "GUIDEVER": (
            "DeepSpyce ver1.0",
            "/ this file was created by DeepSpyce",
        ),
        "FITSVER": ("1.6", "/ FITS definition version"),
    }
)

_header_keys = list(_header_template.keys())

# ============================================================================
# FUNCTIONS
# ============================================================================


def fits_header(header: dict = dict()) -> fits.Header:
    """
    Fits header generator.

    Creates a fits (.fits) header from a dictionary.
    The resulting header entrie can be manipulated as a common dictionary.

    Parameters
    ----------
    header : dict or (str or file like)
        Dictionary used to create .fil header. If a dictionary is not given,
        a path is assumed, and there is an attempt to create a dictionary
        assuming a .iar file like.

    template : bool, default value = False
        Indicates if some common .fits header entries are added (eg. SIMPLE,
        BITPIX, NAXIS...)

    Return
    ----------
    header : fits Header
        Fits Header than can be added into .fits files.
    """
    if not isinstance(header, fits.Header):
        if not isinstance(header, dict):
            header = read_iar(header)
        header = fits.Header(header)
    if template:
        # Sample: TREG_091209.cal.acs.txt [Single Dish FITS (SDFITS)]
        header["SIMPLE"] = ("T", "/ conforms to FITS standard")
        header["BITPIX"] = (8, "/ BITS/PIXEL")
        header["NAXIS"] = (0, "/ number of array dimensions")
        header["EXTEND"] = ("T", "/File contains extensions")
        header["DATE"] = (datetime.today().strftime("%y-%m-%d"), "/")
        header["ORIGIN"] = ("IAR", "/ origin of observation")
        header["TELESCOP"] = ("Antena del IAR", "/ the telescope used")
        header["OBSERVAT"] = ("IAR", "/ the observatory")
        header["GUIDEVER"] = (
            "DeepSpyce ver1.0",
            "/ this file was created by DeepSpyce",
        )
        header["FITSVER"] = ("1.6", "/ FITS definition version")

    return header


def write_fits(
    data: pd.DataFrame,
    outfile: os.PathLike,
    overwrite: bool = False,
    header: dict = dict(),
    name: str = "SINGLE DISH",
):
    """
    Raw file writer.

    Writes a pandas DataFrame (or numpy ndarray),
    data into a .fits file.

    Parameters
    ----------
    data : pandas DataFrame or numpy ndarray
        DataFrame or ndarray to be written into fits file.
    outfile : str or file like
        Path to the .fits file.
    overwrite: bool, default value = False
        Indicates if, in case the outfile already exists, it is overwriten.
    order : {"C", "F"}, default value = "F"
        Write the data elements using this index order.
        "C" indicates C-like index order.
        "F" indicates Fortran-like index order.
    """
    """Create .fits from dataframe or ndarray."""
    hdr = fits_header(header)
    primary_hdu = fits.PrimaryHDU(header=hdr)
    tab = Table(np.asarray(data))
    bintable_hdu = fits.BinTableHDU(tab, header=hdr, name=name)
    hdul = fits.HDUList([primary_hdu, bintable_hdu])
    hdul.writeto(outfile, overwrite=overwrite)

    return

