"""SAX Crossing Models."""

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax

from .ports import PortNamer


@jax.jit
@validate_call
def crossing_ideal(wl: sax.FloatArrayLike = 1.5) -> sax.SDict:
    """Crossing model."""
    one = jnp.ones_like(jnp.asarray(wl))
    p = PortNamer(2, 2)
    return sax.reciprocal(
        {
            (p.o1, p.o3): one,
            (p.o2, p.o4): one,
        }
    )
