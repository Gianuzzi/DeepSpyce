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

from deepspyce import datasets
from deepspyce.utils.files_utils import (
    file_exists,
    is_filelike,
    is_opened,
    is_readable,
    is_writable,
    open_file,
    read_file,
    write_to_file,
)

import pytest

# ============================================================================
# CONSTANTS
# ============================================================================

rawpath = datasets.PATH / "20201027_133329_test.raw"
iarpath = datasets.PATH / "J0437-4715_1_A1.iar"

# =============================================================================
# TESTS
# =============================================================================


def test_file_exists(wrong_path: str):
    assert file_exists(rawpath)
    assert not file_exists(wrong_path)


@pytest.mark.parametrize("data", [0.0, [0], (0, 0.0), {0: 0}])
def test_file_exists_wrong_input(data: any):
    with pytest.raises(TypeError):
        file_exists(data)


def test_open_file_closed():
    with open_file(iarpath) as f1:
        assert not f1.closed
        with open_file(f1):
            assert not f1.closed
    f2 = open_file(f1)
    assert not f2.closed
    f2.close()


def test_open_file_read():
    with open_file(iarpath) as f:
        assert f.readable()
        assert not f.writable()

    assert f.closed


def test_open_file_append():
    with open_file(iarpath, mode="a", overwrite=True) as f:
        assert not f.readable()
        assert f.writable()

    assert f.closed


def test_open_file_wrong_path(wrong_path: str):
    with pytest.raises(FileNotFoundError):
        open_file(wrong_path)


def test_open_file_no_overwrite():
    with pytest.raises(FileExistsError):
        open_file(rawpath, mode="w", overwrite=False)


def test_is_filelike(stream: callable):
    path = stream()

    assert is_filelike(path)


@pytest.mark.parametrize("data", [0.0, [0], (0, 0.0), {0: 0}])
def test_is_filelike_wrong_input(data: any):
    assert not is_filelike(data)


def test_is_opened(stream: callable):
    path = stream()

    assert is_opened(path)
    path.close()
    assert not is_opened(path)


def test_is_readable():
    with open_file(iarpath, mode="r") as f:
        assert is_readable(f)
    with open_file(iarpath, mode="a", overwrite=True) as f:
        assert not is_readable(f)


def test_is_writable():
    with open_file(iarpath, mode="r") as f:
        assert not is_writable(f)
    with open_file(iarpath, mode="a", overwrite=True) as f:
        assert is_writable(f)


def test_read_file():
    f = read_file(iarpath)

    assert isinstance(f, str)
    assert f[:10] == "Source Nam"
    assert f[-10:] == "s,1\nCal,0\n"


def test_read_file_wrong_uft8():
    with pytest.raises(UnicodeDecodeError):
        read_file(rawpath)


def test_read_file_bin():
    f = read_file(rawpath, "rb")

    assert isinstance(f, bytes)
    assert f[:10] == b"\x00\x00\x00\x00\x00\x00\xc0:\x00\x00"
    assert f[-10:] == b"\xd3\xee\x00\x00\x00\x00\x00\x00\x00\x00"


def test_write_to_file_none(stream: callable):
    path = stream()
    ptr = path.tell()
    write_to_file(b"", path)

    assert path.tell() == ptr


def test_write_to_file(namedtempfile: callable):
    tmp = namedtempfile()
    write_to_file(b"42", tmp.name, "wb", overwrite=True)
    with open(tmp.name, "ab") as f:
        assert f.tell() == 2


def test_write_to_file_wrong():
    with pytest.raises(OSError):
        write_to_file(b"42", 123, "wb", overwrite=True)
