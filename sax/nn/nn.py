""" Neural network tools and architectures """

from __future__ import annotations

import jax
import jax.numpy as jnp
from .utils import (
    normalize,
    denormalize,
)
from typing import Optional, Callable, Dict, Tuple, Union
from .._typing import ComplexFloat, ComplexFloat, Array


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


def generate_dense_weights(
    key: Union[int, Array],
    sizes: Tuple[int, ...],
    input_names: Optional[Tuple[str, ...]] = None,
    output_names: Optional[Tuple[str, ...]] = None,
    preprocess=preprocess,
) -> Dict[str, ComplexFloat]:
    """Generate the weights for a dense neural network

    Args:
        key: the random jax.random.PRNGKey or seed to generate the weights with
        sizes: the sizes of the dense neural network weights
        input_names: the names of the input parameters of the neural network.
            If given, the input dimension of the neural network will be derived
            from the number of features after preprocessing these names, i.e. this
            input dimension will then be prepended to `sizes`.
        output_names: the names of the output parameters the neural network
            should predict. If given, the output dimension of the neural
            network will be set to the number of output names, i.e. th number
            of output names will be appended to `sizes`.
        preprocess: the preprocessing function which will be used in the neural
            network. This preprocessing function determines the input dimension
            (if `input_names` is given.)

    Returns:
        the dictionary of random weights and biases.
    """

    if isinstance(key, int):
        key = jax.random.PRNGKey(key)
    assert isinstance(key, jnp.ndarray)

    sizes = tuple(s for s in sizes)
    if input_names:
        arr = preprocess(*jnp.ones(len(input_names)))
        assert isinstance(arr, jnp.ndarray)
        sizes = (arr.shape[-1],) + sizes
    if output_names:
        sizes = sizes + (len(output_names),)

    keys = jax.random.split(key, 2 * len(sizes))
    rand = jax.nn.initializers.lecun_normal()
    weights = {}
    for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        weights[f"w{i}"] = rand(keys[2 * i], (m, n))
        weights[f"b{i}"] = rand(
            keys[2 * i + 1],
            (
                1,
                n,
            ),
        ).ravel()

    return weights
