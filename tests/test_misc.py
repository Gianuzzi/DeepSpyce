#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   DeepSpyce Project (https://github.com/suaraujo/DeepSpyce).
# Copyright (c) 2020, Susana Beatriz Araujo Furlan
# License: MIT
#   Full Text: https://github.com/suaraujo/DeepSpyce/blob/master/LICENSE


# ============================================================================
# IMPORTS
# ============================================================================

import warnings

from deepspyce.utils.misc import (
    _check_key_pos,
    _my_decode,
    bytes_to_data,
    bytes_to_dict,
    check_dict_types,
    data_to_bytes,
    dict_to_bytes,
)

import numpy as np


# =============================================================================
# TESTS
# =============================================================================


def test_my_decode(stream: callable):
    # Test with bytes input
    encoded = b"\x00\x00\x00\x03abc"
    assert _my_decode(encoded) == "abc"

    # Test with io.BytesIO input
    encoded = stream(b"\x00\x00\x00\x03abc")
    assert _my_decode(encoded) == "abc"

    # Test with different format
    encoded = b"B\xf6\xe6f'"
    np.testing.assert_equal(_my_decode(encoded, fmt=">f"), np.float32(123.45))

    # Test with different byte order
    encoded = b"\xcd\xcc\xcc\xcc\xcc\xdc^@"
    np.testing.assert_equal(_my_decode(encoded, fmt="<d"), 123.45)


def test_check_key_pos():
    dicc = {"a": 1, "b": 2, "c": 3}
    key = "b"
    pos = 1
    result = _check_key_pos(dicc, key, pos)
    assert result

    key = "d"
    result = _check_key_pos(dicc, key, pos)
    assert result is None

    key = "a"
    pos = 2
    result = _check_key_pos(dicc, key, pos)
    assert not result


def test_check_dict_types():
    header = {"KEY1": 1, "KEY2": "value", "KEY3": [1, 2, 3]}
    key_fmts = {"KEY1": int, "KEY2": str, "KEY3": list}

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        result = check_dict_types(header, key_fmts)
        assert len(w) == 0
        assert result is True

    header = {"KEY1": 1, "KEY2": "value", "KEY3": "not a list"}

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        result = check_dict_types(header, key_fmts)
        assert len(w) == 1
        assert (
            str(w[0].message)
            == "WARNING. Key KEY3 value should be type <class 'list'>"
        )
        assert result is False

    header = {
        "KEY1": 1,
        "KEY2": "value",
        "KEY3": [1, 2, 3],
        "UNEXPECTED_KEY": "some value",
    }

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        result = check_dict_types(header, key_fmts)
        assert len(w) == 1
        assert str(w[0].message) == "WARNING. Unexpected key: UNEXPECTED_KEY"
        assert result is False


def test_dict_to_bytes():
    header = {"key1": 1, "key2": 2.0}
    key_fmts = {"key1": int, "key2": float}

    encoded = dict_to_bytes(header, key_fmts)
    assert isinstance(encoded, bytes)

    header = {"key1": 1, "key2": "value2"}
    key_fmts = {"key1": int}

    encoded = dict_to_bytes(header, key_fmts)
    assert isinstance(encoded, bytes)

    header = {"HEADER_START": None, "key1": 1, "key2": 2.0, "HEADER_END": None}
    encoded = dict_to_bytes(header)
    assert isinstance(encoded, bytes)


def test_bytes_to_data():
    dtype = np.dtype(">i8")
    data = np.array([1, 2, 3, 4], dtype=dtype)
    encoded = data.tobytes()

    # Test default values
    result = bytes_to_data(encoded)
    assert result.dtype == dtype
    assert result.shape == (4,)
    np.testing.assert_equal(result, data)

    # Test with n_cols = 2
    result = bytes_to_data(encoded, n_cols=2)
    assert result.dtype == dtype
    assert result.shape == (2, 2)
    np.testing.assert_equal(result, data.reshape((2, 2), order="F"))

    # Test with fmt = "<i8"
    dtype = np.dtype("<i8")
    data = np.array([1, 2, 3, 4], dtype=dtype)
    encoded = data.tobytes()

    result = bytes_to_data(encoded, fmt=dtype)
    assert result.dtype == dtype
    np.testing.assert_equal(result, data)

    # Test with swap = True
    result = bytes_to_data(encoded, swap=True)
    assert result.dtype == dtype
    np.testing.assert_equal(result, data)

    # Test with swap = "big"
    result = bytes_to_data(encoded, swap="little")
    assert result.dtype == dtype
    np.testing.assert_equal(result, data)


def test_data_to_bytes():
    # Test with simple input data and default values
    data = np.array([1, 2, 3, 4])
    expected_result = (
        b"\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00"
        + b"\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00"
    )
    result = data_to_bytes(data)
    assert (
        result == expected_result
    ), f"Expected {expected_result}, but got {result}"

    # Test with custom format
    data = np.array([1, 2, 3, 4], dtype="int16")
    expected_result = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    result = data_to_bytes(data, fmt="int16")
    assert (
        result == expected_result
    ), f"Expected {expected_result}, but got {result}"

    # Test with different order
    data = np.array([[1, 2], [3, 4]], dtype="int16")
    expected_result = b"\x01\x00\x02\x00\x03\x00\x04\x00"
    result = data_to_bytes(data, order="C")
    assert (
        result == expected_result
    ), f"Expected {expected_result}, but got {result}"

    # Test with byte swapping
    data = np.array([1, 2, 3, 4], dtype=">i4")
    expected_result = (
        b"?\xf0\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x08\x00"
        + b"\x00\x00\x00\x00\x00@\x10\x00\x00\x00\x00\x00\x00"
    )
    result = data_to_bytes(data, swap=True)
    assert (
        result == expected_result
    ), f"Expected {expected_result}, but got {result}"


def test_bytes_to_dict():
    # Test Case 1: header starts with "HEADER_START"
    encoded = (
        b"\x0c\x00\x00\x00HEADER_START"
        + b"\x04\x00\x00\x00key1\x00\x00\x00\x00\x00\x00 @"
        + b"\n\x00\x00\x00HEADER_END"
    )
    key_fmts = {"key1": float}
    expected_output = {"key1": 8.0}
    assert bytes_to_dict(encoded, key_fmts) == expected_output

    # Test Case 2: header does not start with "HEADER_START"
    encoded = (
        b"\x04\x00\x00\x00key1\x00\x00\x00\x00\x00\x00 @"
        + b"\n\x00\x00\x00HEADER_END"
    )
    key_fmts = {"key1": float}
    expected_output = {"key1": 8.0}
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        assert bytes_to_dict(encoded, key_fmts) == expected_output
        assert len(w) == 1
        assert (
            str(w[0].message)
            == "WARNING. Header does not start with 'HEADER_START'"
        )

    # Test Case 3: key_fmts is not provided
    encoded = (
        b"\x0c\x00\x00\x00HEADER_START"
        + b"\x04\x00\x00\x00key1\x00\x00\x00\x00\x00\x00 @"
        + b"\n\x00\x00\x00HEADER_END"
    )
    expected_output = {"key1": 8.0}
    assert bytes_to_dict(encoded) == expected_output

    # Test Case 4: Swap is True
    encoded = (
        b"\x00\x00\x00\x0cHEADER_START"
        + b"\x00\x00\x00\x04key1@ \x00\x00\x00\x00\x00\x00"
        + b"\x00\x00\x00\nHEADER_END"
    )
    key_fmts = {"key1": float}
    expected_output = {"key1": 8.0}
    assert bytes_to_dict(encoded, key_fmts, True) == expected_output
