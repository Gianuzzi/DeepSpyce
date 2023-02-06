#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   DeepSpyce Project (https://github.com/suaraujo/DeepSpyce).
# Copyright (c) 2020, Susana Beatriz Araujo Furlan
# License: MIT
#   Full Text: https://github.com/suaraujo/DeepSpyce/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""The deepspyce.io module includes I/O tools."""

# =============================================================================
# IMPORTS
# =============================================================================

# from .fits import fits_header, write_fits  # noqa
from .iar import read_iar, write_iar  # noqa
from .filterbank import read_filterbank, read_filterbank_data, read_filterbank_header, write_filterbank  # noqa
from .raw import read_raw, write_raw  # noqa
