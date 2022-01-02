""" Utilities for neural network models """

from __future__ import annotations

from collections import namedtuple

import pandas as pd
import jax.numpy as jnp

from typing import Tuple
from .._typing import ComplexFloat


def cartesian_product(*arrays: ComplexFloat) -> ComplexFloat:
    """calculate the n-dimensional cartesian product

    i.e. create all possible combinations of all elements in a given collection
    of arrays.

    Args:
        *arrays:  the arrays to calculate the cartesian product for

    Returns:
        the cartesian product.
    """
    ixarrays = jnp.ix_(*arrays)
    barrays = jnp.broadcast_arrays(*ixarrays)
    sarrays = jnp.stack(barrays, -1)
    assert isinstance(sarrays, jnp.ndarray)
    product = sarrays.reshape(-1, sarrays.shape[-1])
    assert isinstance(product, jnp.ndarray)
    return product


def denormalize(
    x: ComplexFloat, mean: ComplexFloat = 0.0, std: ComplexFloat = 1.0
) -> ComplexFloat:
    """denormalize an array with a given mean and standard deviation

    Args:
        x: the array to denormalize
        mean: the mean of the denormalized tensor
        std: the standard deviation of the denormalized tensor

    Returns:
        the denormalized tensor
    """
    return x * std + mean


norm = namedtuple("norm", ("mean", "std"))


def get_normalization(x: ComplexFloat):
    """Get mean and standard deviation for a given array

    Args:
        x: the array to get the normalization for

    Return:
        the mean and standard deviation
    """
    if isinstance(x, (complex, float)):
        return x, 0.0
    return norm(x.mean(0), x.std(0))


def get_df_columns(df: pd.DataFrame, *names: str) -> Tuple[ComplexFloat, ...]:
    """Get certain columns from a pandas DataFrame as jax.numpy arrays

    Args:
        df: the dataframe to get the columns from
        names: the names of the dataframe columns to get

    Returns:
        the columns of the dataframe as a namedtuple
    """

    tup = namedtuple("params", names)
    params_list = []
    for name in names:
        column_np = df[name].values
        column_jnp = jnp.array(column_np)
        assert isinstance(column_jnp, jnp.ndarray)
        params_list.append(column_jnp.ravel())
    return tup(*params_list)


def normalize(
    x: ComplexFloat, mean: ComplexFloat = 0.0, std: ComplexFloat = 1.0
) -> ComplexFloat:
    """normalize an array with a given mean and standard deviation

    Args:
        x: the array to normalize
        mean: the mean to normalize the array with
        std: the standard deviation to normalize the array with

    Returns:
        the normalized tensor
    """
    return (x - mean) / std
