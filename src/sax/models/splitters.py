"""SAX Default Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def splitter_ideal(*, coupling: sax.FloatArrayLike = 0.5) -> sax.SDict:
    """A simple coupler model."""
    kappa = jnp.asarray(coupling**0.5)
    tau = jnp.asarray((1 - coupling) ** 0.5)
    return sax.reciprocal(
        {
            ("in0", "out0"): tau,
            ("in0", "out1"): kappa,
        },
    )
