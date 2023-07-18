# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: sax
#     language: python
#     name: sax
# ---

# +
# default_exp nn.utils
# -

# # Utils
#
# > loss functions and utilitites for SAX neural networks

# +
# hide
import matplotlib.pyplot as plt
from fastcore.test import test_eq
from pytest import approx, raises

import os, sys; sys.stderr = open(os.devnull, "w")

# +
# export
from __future__ import annotations

from collections import namedtuple
from typing import Tuple

import jax.numpy as jnp
import pandas as pd
from sax.typing_ import ComplexFloat


# -

# export
def cartesian_product(*arrays: ComplexFloat) -> ComplexFloat:
    """calculate the n-dimensional cartesian product of an arbitrary number of arrays"""
    ixarrays = jnp.ix_(*arrays)
    barrays = jnp.broadcast_arrays(*ixarrays)
    sarrays = jnp.stack(barrays, -1)
    assert isinstance(sarrays, jnp.ndarray)
    product = sarrays.reshape(-1, sarrays.shape[-1])
    assert isinstance(product, jnp.ndarray)
    return product


# export
def denormalize(x: ComplexFloat, mean: ComplexFloat = 0.0, std: ComplexFloat = 1.0) -> ComplexFloat:
    """denormalize an array with a given mean and standard deviation"""
    return x * std + mean


# exporti
norm = namedtuple("norm", ("mean", "std"))


# export
def get_normalization(x: ComplexFloat):
    """Get mean and standard deviation for a given array"""
    if isinstance(x, (complex, float)):
        return x, 0.0
    return norm(x.mean(0), x.std(0))


# export
def get_df_columns(df: pd.DataFrame, *names: str) -> Tuple[ComplexFloat, ...]:
    """Get certain columns from a pandas DataFrame as jax.numpy arrays"""
    tup = namedtuple("params", names)
    params_list = []
    for name in names:
        column_np = df[name].values
        column_jnp = jnp.array(column_np)
        assert isinstance(column_jnp, jnp.ndarray)
        params_list.append(column_jnp.ravel())
    return tup(*params_list)


# export
def normalize(
    x: ComplexFloat, mean: ComplexFloat = 0.0, std: ComplexFloat = 1.0
) -> ComplexFloat:
    """normalize an array with a given mean and standard deviation"""
    return (x - mean) / std
