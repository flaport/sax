"""SAX Crossing Models."""

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def crossing_ideal(wl: sax.FloatArrayLike = 1.5) -> sax.SDict:
    """Crossing model."""
    one = jnp.ones_like(jnp.asarray(wl))
    return sax.reciprocal(
        {
            ("o1", "o3"): one,
            ("o2", "o4"): one,
        }
    )
