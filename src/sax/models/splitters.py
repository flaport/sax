"""SAX Default Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax

from .ports import PortNamer


@jax.jit
@validate_call
def splitter_ideal(*, coupling: sax.FloatArrayLike = 0.5) -> sax.SDict:
    """A simple coupler model."""
    kappa = jnp.asarray(coupling**0.5)
    tau = jnp.asarray((1 - coupling) ** 0.5)

    p = PortNamer(num_inputs=1, num_outputs=2)
    return sax.reciprocal(
        {
            (p.in0, p.out0): tau,
            (p.in0, p.out1): kappa,
        },
    )
