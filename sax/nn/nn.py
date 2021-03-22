""" Neural network tools and architectures """

import jax
import jax.numpy as jnp
from .utils import normalize, denormalize


def preprocess(*params):
    """preprocess parameters

    This function does the following steps
        - all arguments are first casted into the same shape
        - then pairs of arguments are divided into each other to create
              relative arguments.
        - all arguments are then stacked into one big tensor

    Args:
        *params: the parameters to combine into a stacked tensor. Note that all
            these parameters should be broadcastable to the same shape!
    """
    x = jnp.stack(jnp.broadcast_arrays(*params), 0)
    to_concatenate = [x]
    for i in range(1, x.shape[0]):
        _x = jnp.roll(x, shift=i, axis=0)
        to_concatenate.append(x / _x)
        to_concatenate.append(_x / x)
    x = jnp.concatenate(to_concatenate, 0)
    return x


def dense(weights, *params, preprocess=preprocess, activation=jax.nn.leaky_relu):
    """simple dense neural network

    Args:
        weights: the weights of the dense neural network in dictionary format.
            The key's of the dictionary go from 'w0', 'b0' to 'wN', 'bN', with N the
            number of layers.
        *params: the parameters to use as input to the neural network. Note
            that all these parameters should be broadcastable to the same shape!
    """
    x = preprocess(*params)
    x = normalize(x, mean=weights["x_mean"], std=weights["x_std"])
    for i in range(len(weights)):
        x = activation(x @ weights[f"w{i}"] + weights[f"b{i}"])
    yhat = denormalize(x, mean=weights["y_mean"], std=weights["y_std"])
    return yhat


def neff(weights, *params, preprocess=preprocess, activation=jax.nn.leaky_relu):
    """predict the effective index of a waveguide

    Args:
        weights: the weights of the dense neural network in dictionary format.
            The key's of the dictionary go from 'w0', 'b0' to 'wN', 'bN', with N the
            number of layers.
        *params: the relevant parameters for calculating the effective index, such as
            wg_width, wg_height, slab_height, temperature, wavelength, ... Note
            that all these parameters should be broadcastable to the same
            shape!

    Note:
        this function is an alias for ``dense``.
    """
    return dense(weights, *params, preprocess=preprocess, activation=activation)
