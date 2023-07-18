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
# default_exp nn.loss
# -

# # Loss
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

from typing import Dict

import jax.numpy as jnp
from sax.typing_ import ComplexFloat


# +
# export

def mse(x: ComplexFloat, y: ComplexFloat) -> float:
    """mean squared error"""
    return ((x - y) ** 2).mean()


# +
# export

def huber_loss(x: ComplexFloat, y: ComplexFloat, delta: float=0.5) -> float:
    """huber loss"""
    return ((delta ** 2) * ((1.0 + ((x - y) / delta) ** 2) ** 0.5 - 1.0)).mean()


# -

# The huber loss is like the mean squared error close to zero and mean
# absolute error for outliers

# +
# export

def l2_reg(weights: Dict[str, ComplexFloat]) -> float:
    """L2 regularization loss"""
    numel = 0
    loss = 0.0
    for w in (v for k, v in weights.items() if k[0] in ("w", "b")):
        numel = numel + w.size
        loss = loss + (jnp.abs(w) ** 2).sum()
    return loss / numel
