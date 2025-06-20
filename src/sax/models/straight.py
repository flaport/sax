"""SAX Default Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax
from sax.models.ports import PortNamer


@jax.jit
@validate_call
def straight(
    *,
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    neff: sax.FloatArrayLike = 2.34,
    ng: sax.FloatArrayLike = 3.4,
    length: sax.FloatArrayLike = 10.0,
    loss_dB_cm: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    """A simple straight waveguide model."""
    dwl: sax.FloatArray = sax.into[sax.FloatArray](wl) - wl0
    dneff_dwl = (ng - neff) / wl0
    _neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * _neff * length / wl
    amplitude = jnp.asarray(10 ** (-1e-4 * loss_dB_cm * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    p = PortNamer(1, 1)
    return sax.reciprocal(
        {
            (p.in0, p.out0): transmission,
        },
    )


@jax.jit
@validate_call
def attenuator(*, loss: sax.FloatArrayLike = 0.0) -> sax.SDict:
    """Attenuator model."""
    transmission = jnp.asarray(10 ** (-loss / 20), dtype=complex)
    p = PortNamer(1, 1)
    return sax.reciprocal(
        {
            (p.in0, p.out0): transmission,
        }
    )


@jax.jit
@validate_call
def phase_shifter(
    wl: sax.FloatArrayLike = 1.55,
    neff: sax.FloatArrayLike = 2.34,
    voltage: sax.FloatArrayLike = 0,
    length: sax.FloatArrayLike = 10,
    loss: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    """Simple phase shifter model."""
    deltaphi = voltage * jnp.pi
    phase = 2 * jnp.pi * neff * length / wl + deltaphi
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    p = PortNamer(1, 1)
    return sax.reciprocal(
        {
            (p.o1, p.o2): transmission,
        }
    )
