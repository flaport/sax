""" Loss functions for training neural networks """

from __future__ import annotations

import jax.numpy as jnp

from typing import Dict
from .._typing import Array


def mse(x: Array, y: Array) -> float:
    """mean squared error

    Args:
        x: the prediction
        y: the target

    Returns:
        The mean squared error between prediction and target
    """
    return ((x - y) ** 2).mean()


def huber_loss(x: Array, y: Array, delta=0.5) -> float:
    """huber loss

    The huber loss is like the mean squared error close to zero and mean
    absolute error for outliers

    Args:
        x: the prediction
        y: the target
        delta: the parameter controlling where in between MSE and MAE the huber
            loss is. δ=0 corresponds to MAE. δ=1 corresponds to MSE.

    Returns:
        the huber loss between prediction and target

    """
    return ((delta ** 2) * ((1.0 + ((x - y) / delta) ** 2) ** 0.5 - 1.0)).mean()


def l2_reg(weights: Dict[str, Array]) -> float:
    """L2 regularization loss

    Args:
        weights: the neural network weight dictionary. Per convention, the
            weights should be named `w{i}` and the biasses should be named `b{i}`.

    Returns:
        the L2 regularization loss

    """
    numel = 0
    loss = 0.0
    for w in (v for k, v in weights.items() if k[0] in ("w", "b")):
        numel = numel + w.size
        loss = loss + (jnp.abs(w) ** 2).sum()
    return loss / numel
