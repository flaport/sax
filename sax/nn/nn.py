""" Neural network tools and architectures """

from __future__ import annotations

import os
import re
import functools

import jax
import jax.numpy as jnp
from .utils import (
    normalize,
    denormalize,
    get_dense_weights_path,
    get_norm_path,
    load_json,
    norm,
)
from typing import Callable, Dict, Tuple, Optional
from .._typing import ComplexFloat, ComplexFloat, Array

__all__ = ["dense", "preprocess", "load_dense_model"]


def preprocess(*params: ComplexFloat) -> ComplexFloat:
    """preprocess parameters

    - all arguments are first casted into the same shape
    - then pairs of arguments are divided into each other to create relative arguments.
    - all arguments are then stacked into one big tensor

    Args:
        *params: the parameters to combine into a stacked tensor. Note that all
            these parameters should be broadcastable to the same shape!
    """
    x = jnp.stack(jnp.broadcast_arrays(*params), -1)
    assert isinstance(x, jnp.ndarray)
    to_concatenate = [x]
    for i in range(1, x.shape[-1]):
        _x = jnp.roll(x, shift=i, axis=-1)
        to_concatenate.append(x / _x)
        to_concatenate.append(_x / x)
    x = jnp.concatenate(to_concatenate, -1)
    assert isinstance(x, jnp.ndarray)
    return x


def dense(
    weights: Dict[str, Array],
    *params: ComplexFloat,
    x_norm: Tuple[float, float] = (0.0, 1.0),
    y_norm: Tuple[float, float] = (0.0, 1.0),
    preprocess: Callable = preprocess,
    activation: Callable = jax.nn.leaky_relu,
) -> ComplexFloat:
    """simple dense neural network

    Args:
        weights: the weights of the dense neural network in dictionary format.
            The key's of the dictionary should go from 'w0', 'b0' to 'wN', 'bN',
            with N the number of layers.
        *params: the parameters to use as input to the neural network. Note
            that all these parameters should be broadcastable to the same shape!
        x_norm: normalization constants (mean, std) for the input data
        y_norm: normalization constants (mean, std) for the output data
    """
    x_mean, x_std = x_norm
    y_mean, y_std = y_norm
    x = preprocess(*params)
    x = normalize(x, mean=x_mean, std=x_std)
    for i in range(len([w for w in weights if w.startswith("w")])):
        x = activation(x @ weights[f"w{i}"] + weights.get(f"b{i}", 0.0))
    y = denormalize(x, mean=y_mean, std=y_std)
    return y


def load_dense(
    *layer_sizes: int,
    input_names: Optional[Tuple[str, ...]] = None,
    output_names: Optional[Tuple[str, ...]] = None,
) -> Callable:
    """Load a pre-trained dense model

    Args:
        *layer_sizes: the sizes of the dense layers to load
        input_names: the names of the parameters the neural network uses for prediction.
        output_names: the names of the parameters the neural network predicts
    """
    weights_path = get_dense_weights_path(
        *layer_sizes,
        prefix="dense",
        folderpath="weights",
        input_names=input_names,
        output_names=output_names,
    )
    if not os.path.exists(weights_path):
        raise ValueError("Cannot find weights path for given parameters")
    x_norm_path = get_norm_path(
        (layer_sizes[0],), folderpath="norms", names=input_names
    )
    if not os.path.exists(x_norm_path):
        raise ValueError("Cannot find normalization for input parameters")
    y_norm_path = get_norm_path(
        (layer_sizes[-1],), folderpath="norms", names=output_names
    )
    if not os.path.exists(x_norm_path):
        raise ValueError("Cannot find normalization for output parameters")
    weights = load_json(weights_path)
    x_norm_dict = load_json(x_norm_path)
    y_norm_dict = load_json(y_norm_path)
    x_norm = norm(x_norm_dict["mean"], x_norm_dict["std"])
    y_norm = norm(y_norm_dict["mean"], y_norm_dict["std"])
    partial_dense = _PartialDense(weights, x_norm, y_norm, input_names, output_names)
    return partial_dense


class _PartialDense:
    def __init__(self, weights, x_norm, y_norm, input_names, output_names):
        self.weights = weights
        self.x_norm = x_norm
        self.y_norm = y_norm
        self.input_names = input_names
        self.output_names = output_names

    def __call__(self, *params: ComplexFloat) -> ComplexFloat:
        return dense(self.weights, *params, x_norm=self.x_norm, y_norm=self.y_norm)

    def __repr__(self):
        return f"{self.__class__.__name__}{repr(self.input_names)}->{repr(self.output_names)}"
