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

"""Module with header functions."""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings
from datetime import datetime

from astropy.io import fits

from deepspyce.utils.misc import _check_key_pos, dict_to_bytes

from .iar import read_iar

# =============================================================================
# TEMPLATES
# =============================================================================

IAR_HEADER_TYPES = dict(
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

FILT_HEADER_TYPES = dict(
    {
        "telescope_id": int,
        "machine_id": int,
        "data_type": int,
        "rawdatafile": str,
        "source_name": str,
        "barycentric": int,
        "pulsarcentric": int,
        "az_start": float,
        "za_start": float,
        "src_raj": float,
        "src_dej": float,
        "tstart": float,
        "tsamp": float,
        "nbits": int,
        "fch1": float,
        "foff": float,
        "nchans": int,
        "nifs": int,
        "refdm": float,
        "period": float,
    }
)

FITS_HEADER_TEMP = dict(
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

# ============================================================================
# HEADERS
# ============================================================================


def fits_header(header: dict = dict(), template: bool = False) -> fits.Header:
    """
    Fits header generator.

    Creates a fits (.fits) header from a dictionary.
    The output header can be manipulated as a common dictionary.

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
    fitsh : fits Header
        Fits Header than can be added into .fits files.
    """
    if not isinstance(header, fits.Header):
        if not isinstance(header, dict):
            header = read_iar(header)
        fitsh = fits.Header(header)
    else:
        fitsh = header.copy()
    if template:
        for key, value in FITS_HEADER_TEMP.items():
            fitsh[key] = value
        fitsh["DATE"] = (datetime.today().strftime("%y-%m-%d"), "/")

    return fitsh


def filterbank_header(header: dict = dict()) -> dict:
    """
    Filterbank header initializer.

    Creates a None value header, with standard filterbank keys.
    If header is a dictionary, is is used to extend and update (give a value)
    to the created header. The values must be casted into their proper type.

    Parameters
    ----------
    header : dict, default value = dict()
        Dictionary used to initialize update filterbank header.

    Return
    ------
    filth : ``DeepHeader class`` object.
        Dictionary with main filterbank keys.
    """
    filth = dict({k: None for k in FILT_HEADER_TYPES.keys()})
    filth.update(header)

    return filth


def iar_header(header: dict = dict()) -> dict:
    """
    Iar header initializer.

    Creates a None value header, with standard IAR keys.
    If header is a dictionary, is is used to extend and update (give a value)
    to the created header.

    Parameters
    ----------
    header : dict, default value = dict()
        Dictionary used to initialize update iar header.

    Return
    ------
    iarh : ``DeepHeader class`` object.
        Dictionary with main filterbank keys.
    """
    iarh = dict({k: None for k in IAR_HEADER_TYPES.keys()})
    iarh.update(header)

    return iarh


# =============================================================================
# FUNCTIONS
# =============================================================================


def iarh_to_filth(
    iarh: dict, extra: bool = False, encode: bool = False
) -> dict:
    """
    Iar header to filterbank header conversion.

    Builds template dictionary header for filterbank file,
    from an IAR dictionary (matches some entries).

    Parameters
    ----------
    iarh : dict
        Dictionary containing IAR data to be transform info
        filterbank header dict.
    extra : bool, default value = False
        Indicates if some extra keys are added into
        the header (in development).
    encode : Encode the filterbank header output.

    Return
    -------
    filth : dict
        Dictionary with main .fil header entries.
    """

    source_name = iarh.get("Source Name", "None")
    source_ra = iarh.get("Source RA (hhmmss.s)", 0.0)
    source_dec = iarh.get("Source DEC (ddmmss.s)", 0.0)
    telescope_id = int(iarh.get("Telescope ID", 0))
    machine_id = int(iarh.get("Machine ID", 0))
    data_type = int(iarh.get("Data Type", 1))
    avg_data = int(iarh.get("Average Data", 0))
    sub_bands = int(iarh.get("Sub Bands", 0))
    ref_dm = iarh.get("Reference DM", 0.0)

    # ---- ROACH ----
    # THESE MIGHT BE DIFFERENT FOR OTHER RECIEVERS
    # values
    fft_pts = 128
    adc_clk = 200e6
    # parameters
    tsamp = avg_data * fft_pts / adc_clk
    f_off = adc_clk / fft_pts * 1e-6

    time_now = datetime.now().strftime("_%Y%m%d_%H%M%S")

    # tsamp = 1e6 / float(bandwidth) * avg_data
    rawdatafile = f"ds{avg_data}_{source_name}{time_now}.fil"

    filth = {
        "telescope_id": telescope_id,
        "machine_id": machine_id,
        "data_type": data_type,
        "rawdatafile": rawdatafile,
        "source_name": source_name,
        "az_start": 0.0,
        "za_start": 0.0,
        "src_raj": source_ra,
        "src_dej": source_dec,
        "tstart": 0.0,
        "tsamp": tsamp,
        "fch1": 0.0,
        "foff": f_off,
        "nchans": sub_bands,
        "nifs": 1,
        "ibeam": 1,
        "nbeams": 1,
        "refdm": ref_dm,
    }

    if extra:
        pul_period = iarh.get("Pulsar Period", 0.0)
        high_freq = iarh.get("Highest Observation Frequency (MHz)", 0.0)
        observing_time = int(iarh.get("Observing Time (minutes)", 0))
        # gain = iarh.get("Gain (dB)", 0.0)
        bandwidth = int(iarh.get("Total Bandwith (MHz)", 0))

        filth["pul_period"] = pul_period
        filth["high_freq"] = high_freq
        filth["observing_time"] = observing_time
        # fil_header["gain"] = gain
        filth["bandwidth"] = bandwidth

    if encode:
        filth = fixed_header_start_end(filth)
        return dict_to_bytes(filth)

    return filth


# ============================================================================
# UTILS
# ============================================================================


def check_header_start_end(header: dict, verb: bool = False) -> tuple:
    """
    HEADER_START and HEADER_END checker.

    Checks the HEADER_START and HEADER_END entries of the header.
    HEADER_START must be the first entry, and HEADER_END must be the last one.
    Both of them must have None value.

    Parameters
    ----------
    header : dict
        Dictionary to be checked.
    verb : bool, default value = False
        Indicates if a warning or message is raised.

    Return
    -------
    checks : tuple
        Tuple with info about each header START and END.
        (True, True) -> Both right.
        (True, False) -> START right, END wrong.
        (False, True) -> START wrong, END right.
        (False, False) -> Both wrong.
    """
    start = _check_key_pos(header, "HEADER_START", 0, verb)
    end = _check_key_pos(header, "HEADER_END", len(header) - 1, verb)
    if start and (header["HEADER_START"] is not None):
        if verb:
            warnings.warn("header['HEADER_START'] should be None!")
        start = False
    if end and (header["HEADER_END"] is not None):
        if verb:
            warnings.warn("header['HEADER_END'] should be None!")
        end = False
    if (start and end) and verb:
        print("HEADER_START and HEADER END are OK!.")

    return (start, end)


def fixed_header_start_end(header: dict, check: bool = True) -> dict:
    """
    HEADER_START and HEADER_END fixer.

    Fixes the HEADER_START and HEADER_END entries of the header.
    HEADER_START must be the first entry, and HEADER_END must be the last one.
    Both of them must have None value.

    Parameters
    ----------
    header : dict
        Dictionary to be fixed.
    check : bool, default value = True
        Indicates if the header dict START and END are checked before fixing.
        If True and OK, same dict is returned.

    Return
    ----------
    fixedheader : dict
        Fixed START and END header dictionary.
    """
    if check:
        ok = check_header_start_end(header, False)
        if all(ok):
            return header
    fixedheader = dict({"HEADER_START": None})
    for key, value in header.items():
        if key not in ["HEADER_START", "HEADER_END"]:
            fixedheader[key] = value
    fixedheader["HEADER_END"] = None

    return fixedheader
