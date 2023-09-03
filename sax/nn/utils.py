""" Sax Neural Network Default Utilities """

from __future__ import annotations

from collections import namedtuple
from typing import Tuple

import jax.numpy as jnp
import pandas as pd

from ..saxtypes import ComplexArrayND


def cartesian_product(*arrays: ComplexArrayND) -> ComplexArrayND:
    """calculate the n-dimensional cartesian product of an arbitrary number of arrays"""
    ixarrays = jnp.ix_(*arrays)
    barrays = jnp.broadcast_arrays(*ixarrays)
    sarrays = jnp.stack(barrays, -1)
    assert isinstance(sarrays, jnp.ndarray)
    product = sarrays.reshape(-1, sarrays.shape[-1])
    assert isinstance(product, jnp.ndarray)
    return product


def denormalize(
    x: ComplexArrayND, mean: float = 0.0, std: float = 1.0
) -> ComplexArrayND:
    """denormalize an array with a given mean and standard deviation"""
    return x * std + mean


norm = namedtuple("norm", ("mean", "std"))


def get_normalization(x: ComplexArrayND):
    """Get mean and standard deviation for a given array"""
    if isinstance(x, (complex, float)):
        return x, 0.0
    return norm(x.mean(0), x.std(0))


def get_df_columns(df: pd.DataFrame, *names: str) -> Tuple[ComplexArrayND, ...]:
    """Get certain columns from a pandas DataFrame as jax.numpy arrays"""
    tup = namedtuple("params", names)
    params_list = []
    for name in names:
        column_np = df[name].values
        column_jnp = jnp.array(column_np)
        assert isinstance(column_jnp, jnp.ndarray)
        params_list.append(column_jnp.ravel())
    return tup(*params_list)


def normalize(x: ComplexArrayND, mean: float = 0.0, std: float = 1.0) -> ComplexArrayND:
    """normalize an array with a given mean and standard deviation"""
    return (x - mean) / std
