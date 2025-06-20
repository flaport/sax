"""SAX Bend Models."""

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax

from .straight import straight


@jax.jit
@validate_call
def bend(
    wl: sax.FloatArrayLike = 1.5,
    length: sax.FloatArrayLike = 20.0,
    loss: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    """Returns bend Sparameters."""
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    return {k: amplitude * v for k, v in straight(wl=wl, length=length).items()}
