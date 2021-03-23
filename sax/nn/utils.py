""" Utilities for neural network models """

from __future__ import annotations

import os
import re
import json
from collections import namedtuple

import pandas as pd
import jax
import jax.numpy as jnp

from typing import Dict, Union, Tuple, Optional, List
from .._typing import Array, ComplexFloat

__all__ = [
    "cartesian_product",
    "denormalize",
    "generate_random_weights",
    "get_available_hidden_sizes",
    "get_normalization",
    "get_dense_weights_path",
    "get_norm_path",
    "get_df_columns",
    "load_json",
    "norm",
    "normalize",
    "save_json",
]


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


def generate_random_weights(
    *layer_sizes: int,
    key: Union[int, Array] = 42,
) -> Dict[str, ComplexFloat]:
    """Generate the weights for a dense neural network

    Args:
        *layer_sizes: the shapes of the layers
        key: the random PRNGKey or seed to generate the weights with

    Returns:
        the dictionary of random weights and biases.
    """

    if isinstance(key, int):
        key = jax.random.PRNGKey(key)
    assert isinstance(key, jnp.ndarray)
    keys = jax.random.split(key, 2 * len(layer_sizes))
    rand = jax.nn.initializers.lecun_normal()
    weights = {}
    for i, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        weights[f"w{i}"] = rand(keys[2 * i], (m, n))
        weights[f"b{i}"] = rand(keys[2 * i + 1], (1, n))
    return weights


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


def get_layer_sizes(weights: Dict[str, ComplexFloat]) -> Tuple[int, ...]:
    """Get the layer shapes for a given weights dictionary

    Args:
        weights: the weights to get the layer shapes for

    Returns:
        the layer shapes
    """
    layer_sizes = []
    w = None
    for k, w in weights.items():
        if not k.startswith("w"):
            continue
        w = jnp.atleast_1d(jnp.array(w))
        assert isinstance(w, jnp.ndarray)
        layer_sizes.append(w.shape[0])
    if isinstance(w, jnp.ndarray):
        layer_sizes.append(w.shape[1])
    return tuple(layer_sizes)


def get_dense_weights_path(
    *layer_sizes: int,
    prefix: str = "dense",
    folderpath: str = "weights",
    input_names: Optional[Tuple[str, ...]] = None,
    output_names: Optional[Tuple[str, ...]] = None,
):
    """Create the SAX conventional path for a given dictionary

    Args:
        layer_sizes: the layer shapes of the weights to save.
        prefix: the prefix to give the weights filename
        folderpath: the folder to save the weights to
        input_names: the input feature names
        outptut_names: the output (predicted) feature names
    """
    path = os.path.abspath(os.path.join(folderpath, prefix))
    if input_names:
        path = f"{path}-{'-'.join(input_names)}"
    if layer_sizes:
        path = f"{path}-{'x'.join(str(s) for s in layer_sizes)}"
    if output_names:
        path = f"{path}-{'-'.join(output_names)}"
    return f"{path}.json"


def get_available_hidden_sizes(
    prefix: str,
    folderpath: str,
    input_names: Tuple[str, ...],
    output_names: Tuple[str, ...],
) -> List[Tuple[int, ...]]:
    """Get all available hidden sizes given filename parameters

    Args:
        prefix: the prefix of the filenames to check
        folderpath: the folder within to check for matching files
        input_names: the input feature names
        outptut_names: the output (predicted) feature names

    Returns:
        the possible hidden sizes
    """
    all_weightfiles = os.listdir(folderpath)
    possible_weightfiles = (
        s for s in all_weightfiles if s.endswith(f"-{'-'.join(output_names)}.json")
    )
    possible_weightfiles = (
        s
        for s in possible_weightfiles
        if s.startswith(f"{prefix}-{'-'.join(input_names)}")
    )
    possible_weightfiles = (re.sub("[^0-9x]", "", s) for s in possible_weightfiles)
    possible_weightfiles = (re.sub("^x*", "", s) for s in possible_weightfiles)
    possible_weightfiles = (re.sub("x[^0-9]*$", "", s) for s in possible_weightfiles)
    possible_hidden_sizes = (s.strip() for s in possible_weightfiles if s.strip())
    possible_hidden_sizes = (
        tuple(hs.strip() for hs in s.split("x") if hs.strip())
        for s in possible_hidden_sizes
    )
    possible_hidden_sizes = (
        tuple(int(hs) for hs in s[1:-1]) for s in possible_hidden_sizes if len(s) > 2
    )
    possible_hidden_sizes = sorted(
        possible_hidden_sizes, key=lambda hs: (len(hs), max(hs))
    )
    return possible_hidden_sizes


def get_norm_path(
    norm_shape,
    prefix: str = "norm",
    folderpath: str = "norms",
    names: Optional[Tuple[str, ...]] = None,
):
    """Create the SAX conventional path for the normalization constants

    Args:
        norm_shape: the shape of the norm to create a name for
        prefix: the prefix to give the weights filename
        folderpath: the folder to save the weights to
        names: the names of the features in the norm
    """
    path = os.path.abspath(os.path.join(folderpath, prefix))
    if norm_shape:
        path = f"{path}-{'x'.join(str(s) for s in norm_shape)}"
    if names:
        path = f"{path}-{'-'.join(names)}"
    return f"{path}.json"


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


def load_json(path: str) -> Dict[str, ComplexFloat]:
    """Load json weights from given path

    Args:
        path: the path to load the json weights from

    Returns:
        a dictionary of weights
    """
    path = os.path.abspath(os.path.expanduser(path))
    weights = {}
    if os.path.exists(path):
        with open(path, "r") as file:
            for k, v in json.load(file).items():
                _v = jnp.array(v, dtype=float)
                assert isinstance(_v, jnp.ndarray)
                weights[k] = _v
    return weights


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


def save_json(weights: Dict[str, ComplexFloat], path: str):
    """Save json weights to given path

    Args:
        weights: dictionary of weights to save as json
        path: the path to load the json weights from

    """
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        _weights = {}
        for k, v in weights.items():
            v = jnp.atleast_1d(jnp.array(v))
            assert isinstance(v, jnp.ndarray)
            _weights[k] = v.tolist()
        json.dump(_weights, file)
