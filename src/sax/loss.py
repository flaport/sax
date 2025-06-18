"""SAX Loss Functions."""

from typing import cast

import jax.numpy as jnp

from .saxtypes import ComplexArray

__all__ = [
    "huber_loss",
    "l2_reg",
    "mse",
]


def mse(x: ComplexArray, y: ComplexArray) -> float:
    """Mean squared error."""
    return cast(float, (abs(x - y) ** 2).mean())


def huber_loss(x: ComplexArray, y: ComplexArray, delta: float = 0.5) -> float:
    """Huber loss."""
    loss = ((delta**2) * ((1.0 + (abs(x - y) / delta) ** 2) ** 0.5 - 1.0)).mean()
    return cast(float, loss)


def l2_reg(weights: dict[str, ComplexArray]) -> float:
    """L2 regularization loss."""
    numel = 0
    loss = 0.0
    for w in (v for k, v in weights.items() if k[0] in ("w", "b")):
        numel = numel + w.size
        loss = loss + (jnp.abs(w) ** 2).sum()
    return cast(float, loss / numel)
