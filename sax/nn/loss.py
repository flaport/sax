""" SAX Loss Functions """

from __future__ import annotations

from typing import Dict, cast

import jax.numpy as jnp

from ..saxtypes import ComplexArrayND


def mse(x: ComplexArrayND, y: ComplexArrayND) -> float:
    """mean squared error"""
    return cast(float, (abs(x - y) ** 2).mean())


def huber_loss(x: ComplexArrayND, y: ComplexArrayND, delta: float = 0.5) -> float:
    """huber loss"""
    return cast(
        float, ((delta**2) * ((1.0 + (abs(x - y) / delta) ** 2) ** 0.5 - 1.0)).mean()
    )


def l2_reg(weights: Dict[str, ComplexArrayND]) -> float:
    """L2 regularization loss"""
    numel = 0
    loss = 0.0
    for w in (v for k, v in weights.items() if k[0] in ("w", "b")):
        numel = numel + w.size
        loss = loss + (jnp.abs(w) ** 2).sum()
    return loss / numel
