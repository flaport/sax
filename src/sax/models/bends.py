"""SAX Bend Models."""

import jax
from pydantic import validate_call

import sax

from .straight import straight


@jax.jit
@validate_call
def bend(
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    neff: sax.FloatArrayLike = 2.34,
    ng: sax.FloatArrayLike = 3.4,
    length: sax.FloatArrayLike = 10.0,
    loss_dB_cm: sax.FloatArrayLike = 0.1,
) -> sax.SDict:
    """Simple bend model."""
    return straight(
        wl=wl,
        wl0=wl0,
        neff=neff,
        ng=ng,
        length=length,
        loss_dB_cm=loss_dB_cm,
    )
