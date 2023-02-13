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

"""File containing main classes."""

# =============================================================================
# IMPORTS
# =============================================================================

import attr

import numpy as np

import pandas as pd


@attr.s(frozen=True, repr=False)
class DeepHeader(dict):
    """Initialize a dict header to which we can apply function a."""

    header = attr.ib(validator=attr.validators.instance_of(dict))


@attr.s(frozen=True, repr=False)
class DeepFrame:
    """Initialize a dataframe data_df to which we can apply function a."""

    data = attr.ib(
        default=None,
        validator=attr.validators.instance_of((pd.DataFrame, np.ndarray)),
    )
    header = attr.ib(
        default=None, validator=attr.validators.instance_of(DeepHeader)
    )

    # plot_cls = attr.ib()
    # plot = attr.ib(init=False)

    # @plot.default
    # def _plot_default(self):
    #     return self.plot_cls(self)

    # def __getitem__(self, slice):
    #     """x[y] <==> x.__getitem__(y)."""
    #     sliced = self.df.__getitem__(slice)
    #     return DeepFrame(
    #         df=sliced, plot_cls=self.plot_cls, metadata=self.metadata,
    #     )

    # def __dir__(self):
    #     """dir(pdf) <==> pdf.__dir__()."""
    #     return super().__dir__() + dir(self.df)

    # def __getattr__(self, a):
    #     """getattr(x, y) <==> x.__getattr__(y) <==> getattr(x, y)."""
    #     return getattr(self.df, a)

    # def __repr__(self):
    #     """repr(x) <=> x.__repr__()."""
    #     with pd.option_context("display.show_dimensions", False):
    #         df_body = repr(self.df).splitlines()
    #     df_dim = list(self.df.shape)
    #     sdf_dim = f"{df_dim[0]} rows x {df_dim[1]} columns"

    #     fotter = f"\nDeepFrame - {sdf_dim}"
    #     deepframe_data_repr = "\n".join(df_body + [fotter])
    #     return deepframe_data_repr

    # def _repr_html_(self):
    #     ad_id = id(self)

    #     with pd.option_context("display.show_dimensions", False):
    #         df_html = self.df._repr_html_()

    #     rows = f"{self.df.shape[0]} rows"
    #     columns = f"{self.df.shape[1]} columns"

    #     footer = f"DeepFrame - {rows} x {columns}"

    #     parts = [
    #         f'<div class="deepspyce-data-container" id={ad_id}>',
    #         df_html,
    #         footer,
    #         "</div>",
    #     ]

    #     html = "".join(parts)
    #     return html


# ============================================================================
# FUNCTIONS
# ============================================================================


def mk_deepframe(data: np.ndarray, header: dict = dict()):
    """
    Deepframe builder.

    This function builds a deepframe object from a data ndarray
    and a dict header.

    Parameters
    ----------
    data : np.ndarray
        Data to be stored.
    header : dict, default value = dict()
        Header of the data, in dictionary form.

    Return
    ------
    deepframe: ``DeepFrame class`` object.
    """
    return DeepFrame(data=data, header=mk_deepheader(header))


def mk_deepheader(header: dict = dict()):
    """
    Deepheader builder.

    This function builds a deepheader object from a dict header.

    Parameters
    ----------
    header : dict, default value = dict()
        Header of the data, in dictionary form.

    Return
    ------
    deepheader: ``DeepHeader class`` object.
    """
    return DeepHeader(header=header)


# def fourier1D(df: pd.DataFrame, cols: str = None, axis: int = 0, **kwargs):
#     """Compute the 1D discrete Fourier Transform for real input.
#     Parameters
#     ----------
#     df: ``pandas.DataFrame``
#         The dataframe with the values
#     col: ``str``|``list``, optional (default=ALL)
#         Column names or positions to which apply
#         the FFT. If None,
#     axis: ``int``, optional (default=0)
#         Axis over which to compute the FFT.
#     kwargs: ``dict``, optional (default=None)
#         Any extra np.fft.rfft kwargs to be used.
#     Return
#     ------
#     fft: ndarray
#         Numpy array containing the fft of the data.
#     """

#     if cols is None:
#         cols = df.columns
#     fft = np.fft.rfft(df[cols].values, axis=axis, **kwargs)

#     return fft


# def freqs(
#     df: pd.DataFrame,
#     time_step: float = 1.0,
#     axis: int = 0,
#     shift: bool = False,
# ):
#     """Return the 1D Discrete Fourier Transform sample frequencies.
#     Parameters
#     ----------
#     df: ``pandas.DataFrame``
#         The dataframe with the size of the resulting frequencies.
#     time_step: ``float``, optional (default=1.)
#         Sample spacing (inverse of the sampling rate).
#     axis: ``int``, optional (default=0)
#         Axis over which to compute the frequencies.
#     shift: ``bool``, optional (default=False)
#         Shift the zero-frequency component to the center of the spectrum.
#     Return
#     ------
#     freqs: ndarray
#         Numpy 1D array containing the freqs.
#     """

#     freqs = np.fft.rfftfreq(df.shape[axis], time_step)

#     if shift:
#         freqs = np.fft.fftshift(freqs)

#     return freqs


# def power_spectra(df: pd.DataFrame, cols: str = None, **kwargs):
#     """Compute the 1D Power Spectra for real input.
#     Parameters
#     ----------
#     df: ``pandas.DataFrame``
#         The dataframe with the values
#     col: ``str``|``list``, optional (default=ALL)
#         Column names or positions to which calculate
#         the Power Spectra.
#     kwargs: ``dict``, optional (default=None)
#         np.fft.rfft kwargs to be used.
#     Return
#     ------
#     ps: ndarray
#         Numpy array containing the power spectra of the data.
#     """

#     ps = np.abs(fourier1D(df, cols, **kwargs)) ** 2

#     return ps
