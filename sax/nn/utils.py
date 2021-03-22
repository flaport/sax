""" Utilities for neural network models """

from __future__ import annotations

import os
import json

import jax
import jax.numpy as jnp

from typing import Dict, Union
from .._typing import Array, ComplexFloat

__all__ = [
    "cartesian_product",
    "denormalize",
    "generate_random_weights",
    "get_normalization",
    "load_json_weights",
    "normalize",
    "save_json_weights",
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
    key: Union[int, Array], *layer_shapes: int
) -> Dict[str, ComplexFloat]:
    """Generate the weights for a dense neural network

    Args:
        key: the random PRNGKey or seed to generate the weights with
        *layer_shapes: the shapes of the layers

    Returns:
        the dictionary of random weights and biases.
    """

    if isinstance(key, int):
        key = jax.random.PRNGKey(key)
    assert isinstance(key, jnp.ndarray)
    keys = jax.random.split(key, 2 * len(layer_shapes))
    rand = jax.nn.initializers.lecun_normal()
    weights = {}
    for i, (m, n) in enumerate(zip(layer_shapes[:-1], layer_shapes[1:])):
        weights[f"w{i}"] = rand(keys[2 * i], (m, n))
        weights[f"b{i}"] = rand(keys[2 * i + 1], (1, n))
    return weights


def get_normalization(x: ComplexFloat):
    """Get mean and standard deviation for a given array

    Args:
        x: the array to get the normalization for

    Return:
        the mean and standard deviation
    """
    if isinstance(x, (complex, float)):
        return x, 0.0
    return x.mean(0, keepdims=True), x.std(0, keepdims=True)


def load_json_weights(path: str) -> Dict[str, ComplexFloat]:
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


def save_json_weights(weights: Dict[str, ComplexFloat], path: str):
    """Save json weights to given path

    Args:
        weights: dictionary of weights to save as json
        path: the path to load the json weights from

    """
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "w") as file:
        _weights = {}
        for k, v in weights.items():
            v = jnp.atleast_1d(jnp.array(v))
            assert isinstance(v, jnp.ndarray)
            _weights[k] = v.tolist()
        json.dump(_weights, file)
