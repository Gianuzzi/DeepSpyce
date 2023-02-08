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


import struct
import warnings
from collections import OrderedDict
from datetime import datetime

from astropy.io import fits

from .iar import read_iar

# =============================================================================
# UTILS
# =============================================================================

FITS_HEADER_TEMP = OrderedDict(
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


def header_to_bytes(hedicc: dict) -> bytes:
    """
    Dictionary encoder.

    Encodes a dictionary, as a filterbank header.

    Return
    -------
    bina : bytes
        Dictionary encoded into bytes (filterbank header like).
    """
    bina = b""
    for key, value in hedicc.items():
        ret = struct.pack("I", len(key)) + key.encode()
        if value is not None:
            if isinstance(value, str):
                ret = ret + struct.pack("I", len(value)) + value.encode()
            elif isinstance(value, int):
                ret = ret + struct.pack("<l", value)
            else:
                ret = ret + struct.pack("<d", value)
        bina = bina + ret

    return bina


def _iardict_to_fil_header(iardic: dict, extra: bool = False) -> dict:
    """
    Filterbank template header generator.

    Builds template dictionary header for filterbank file,
    from IAR dictionary (because it matches some entries).

    Parameters
    ----------
    iardic : dict
        Dictionary containing IAR data to be transform info
        filterbanck header dict.
    extra : bool, default value = False
        Indicates if some extra keys are added into
        the header (in development).

    Return
    -------
    fil_header : dict
        Dictionary with main .fil header entries.
    """

    source_name = iardic.get("Source Name", "None")
    source_ra = iardic.get("Source RA (hhmmss.s)", 0.0)
    source_dec = iardic.get("Source DEC (ddmmss.s)", 0.0)
    telescope_id = int(iardic.get("Telescope ID", 0))
    machine_id = int(iardic.get("Machine ID", 0))
    data_type = int(iardic.get("Data Type", 1))
    avg_data = int(iardic.get("Average Data", 0))
    sub_bands = int(iardic.get("Sub Bands", 0))
    ref_dm = iardic.get("Reference DM", 0.0)

    # ---- ROACH ----
    # values
    fft_pts = 128
    adc_clk = 200e6
    #  parameters
    tsamp = avg_data * fft_pts / adc_clk
    f_off = adc_clk / fft_pts * 1e-6

    time_now = datetime.now().strftime("_%Y%m%d_%H%M%S")

    # tsamp = 1e6 / float(bandwidth) * avg_data
    rawdatafile = f"ds{avg_data}_{source_name}{time_now}.fil"

    fil_header = {
        "HEADER_START": None,
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
        "HEADER_END": None,
    }

    if extra:
        pul_period = iardic.get("Pulsar Period", 0.0)
        high_freq = iardic.get("Highest Observation Frequency (MHz)", 0.0)
        observing_time = int(iardic.get("Observing Time (minutes)", 0))
        # gain = iardic.get("Gain (dB)", 0.0)
        bandwidth = int(iardic.get("Total Bandwith (MHz)", 0))

        fil_header["pul_period"] = pul_period
        fil_header["high_freq"] = high_freq
        fil_header["observing_time"] = observing_time
        # fil_header["gain"] = gain
        fil_header["bandwidth"] = bandwidth

    return fil_header


# ============================================================================
# FUNCTIONS
# ============================================================================



def fits_header(header: dict = dict()) -> fits.Header: ## FITS
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

def filterbank_header(header: dict = dict()) -> dict: ## FILTERBANK
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
    fheader : ``DeepHeader class`` object.
        Dictionary with main filterbank keys.
    """
    fheader = dict({k: None for k in _header_types.keys()})
    fheader.update(header)

    return fheader

def check_header(header: dict) -> bool: ## FILTERBANK
    """
    Filterbank header data types checker.

    Checks that the filterbank main entries of a given dictionary header
    have their correct data type.
    Raises a warning for each incorrect data type.

    Parameters
    ----------
    header : dict
        Dictionary to be checked.

    Return
    ------
    good : bool
        Boolean indicating if everything is ok.
    """
    for key, value in header.items():
        if key in _header_types.keys():
            dtype = _header_types.get(key)
            if (value is not None) and not isinstance(value, dtype):
                warnings.warn(
                    "WARNING. Key %s value should be type %s" % (key, dtype)
                )
                good = False
        # elif (key in ["HEADER_START", "HEADER_END"]) and (value is not None):
        #     warnings.warn("WARNING. %s key value should be None" %key)
        #     good = False
        else:
            warnings.warn("WARNING. Unexpected key: %s" % key)
            good = False

    return good

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


def iar_to_fil_header(iar: dict, encode: bool = False) -> bytes:
    """
    Iar data to filterbank header converter.

    Converts .iar file (or iar dict) to filterbank header.

    Parameters
    ----------
    iar : dict or (str or file like)
        Path to IAR file, or dict generated from it.
        If None is provided (or a dict different from a IAR like)
        a .fil template header is returned.

    encode : bool, default value = False
        Indicates if the returned header is encoded.

    Return
    ----------
    filheader : dict
        Header useful for .fil file creation.
    """
    if iar is None:
        iar = dict()
    if not isinstance(iar, dict):
        iar = read_iar(iar)
    filheader = _iardict_to_fil_header(iar)
    filheader = fixed_header_start_end(filheader)
    if encode:
        return header_to_bytes(filheader)

    return filheader


def fil_header(
    header: dict = dict(), template: bool = False, encode: bool = False
) -> dict:
    """
    Dictionary to filterbank (.fil) header converter.

    Convert a dictionary into one compatible with filterbank headers.

    Parameters
    ----------
    header : dict or (str or file like)
        Path to IAR file, or dict generated from it.
        If None is provided (or a dict different from a IAR like)
        a .fil template header is returned.

    encode : bool, default value = False
        Indicates if the returned header is encoded.

    Return
    ----------
    filheader : dict
        Header useful for .fil file creation.
    """
    if not isinstance(header, dict):
        header = read_iar(header)
    filheader = _iardict_to_fil_header(header)
    filheader = fixed_header_start_end(filheader)
    if encode:
        return header_to_bytes(filheader)

    return filheader


def fits_header(header: dict = dict(), template: bool = False) -> fits.Header:
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
