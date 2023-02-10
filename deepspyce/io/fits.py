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

from astropy.io import fits
from astropy.table import Table

import numpy as np

import pandas as pd

from .header import fits_header

# ============================================================================
# FUNCTIONS
# ============================================================================


def write_fits(
    data: pd.DataFrame,
    outfile: os.PathLike,
    overwrite: bool = False,
    header: dict = dict(),
    name: str = "SINGLE DISH",
):
    """
    Fits file writer.

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
    header: dict, default value = dict()
        Fits header or dict to use as .fits file header.
    name: str, default value = "SINGLE DISH"
        Name of the header.
    """
    hdr = fits_header(header)
    primary_hdu = fits.PrimaryHDU(header=hdr)
    tab = Table(np.asarray(data))
    bintable_hdu = fits.BinTableHDU(tab, header=hdr, name=name)
    hdul = fits.HDUList([primary_hdu, bintable_hdu])
    hdul.writeto(outfile, overwrite=overwrite)

    return
