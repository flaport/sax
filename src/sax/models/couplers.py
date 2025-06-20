"""SAX Coupler Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def coupler_ideal(*, coupling: sax.FloatArrayLike = 0.5) -> sax.SDict:
    """A simple coupler model."""
    kappa = jnp.asarray(coupling**0.5)
    tau = jnp.asarray((1 - coupling) ** 0.5)
    return sax.reciprocal(
        {
            ("in0", "out0"): tau,
            ("in0", "out1"): 1j * kappa,
            ("in1", "out0"): 1j * kappa,
            ("in1", "out1"): tau,
        },
    )


@jax.jit
@validate_call
def coupler(
    *,
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    length: sax.FloatArrayLike = 0.0,
    coupling0: sax.FloatArrayLike = 0.2,
    dk1: sax.FloatArrayLike = 1.2435,
    dk2: sax.FloatArrayLike = 5.3022,
    dn: sax.FloatArrayLike = 0.02,
    dn1: sax.FloatArrayLike = 0.1169,
    dn2: sax.FloatArrayLike = 0.4821,
) -> sax.SDict:
    r"""Dispersive coupler model.

    equations adapted from photontorch.
    https://github.com/flaport/photontorch/blob/master/photontorch/components/directionalcouplers.py

    kappa = coupling0 + coupling

    Args:
        wl: wavelength (um).
        wl0: center wavelength (um).
        length: coupling length (um).
        coupling0: bend region coupling coefficient from FDTD simulations.
        dk1: first derivative of coupling0 vs wavelength.
        dk2: second derivative of coupling vs wavelength.
        dn: effective index difference between even and odd modes.
        dn1: first derivative of effective index difference vs wavelength.
        dn2: second derivative of effective index difference vs wavelength.

    .. code::

          coupling0/2        coupling        coupling0/2
        <-------------><--------------------><---------->
         o2 ________                           _______o3
                    \                         /
                     \        length         /
                      =======================
                     /                       \
            ________/                         \________
         o1                                           o4

                      ------------------------> K (coupled power)
                     /
                    / K
           -----------------------------------> T = 1 - K (transmitted power)
    """
    dwl = wl - wl0
    dn = dn + dn1 * dwl + 0.5 * dn2 * dwl**2
    kappa0 = coupling0 + dk1 * dwl + 0.5 * dk2 * dwl**2
    kappa1 = jnp.pi * dn / wl

    tau = jnp.cos(kappa0 + kappa1 * length)
    kappa = -jnp.sin(kappa0 + kappa1 * length)
    return sax.reciprocal(
        {
            ("in0", "out0"): tau,
            ("in0", "out1"): 1j * kappa,
            ("in1", "out0"): 1j * kappa,
            ("in1", "out1"): tau,
        }
    )


@jax.jit
@validate_call
def grating_coupler(
    *,
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    loss: sax.FloatArrayLike = 0.0,
    reflection: sax.FloatArrayLike = 0.0,
    reflection_fiber: sax.FloatArrayLike = 0.0,
    bandwidth: sax.FloatArrayLike = 40e-3,
) -> sax.SDict:
    """Grating_coupler model.

    equation adapted from photontorch grating coupler
    https://github.com/flaport/photontorch/blob/master/photontorch/components/gratingcouplers.py

    Args:
        wl: wavelength.
        wl0: center wavelength.
        loss: in dB.
        reflection: from waveguide side.
        reflection_fiber: from fiber side.
        bandwidth: 3dB bandwidth (um).

    .. code::

                       fiber out0

                    /  /  /  /
                   /  /  /  /

                 _|-|_|-|_|-|___
            in0  ______________|

    """
    one = jnp.ones_like(wl)
    reflection = jnp.asarray(reflection) * one
    reflection_fiber = jnp.asarray(reflection_fiber) * one
    amplitude = jnp.asarray(10 ** (-loss / 20))
    sigma = jnp.asarray(bandwidth / (2 * jnp.sqrt(2 * jnp.log(2))))
    transmission = jnp.asarray(amplitude * jnp.exp(-((wl - wl0) ** 2) / (2 * sigma**2)))
    return sax.reciprocal(
        {
            ("in0", "in0"): reflection,
            ("in0", "out0"): transmission,
            ("out0", "in0"): transmission,
            ("out0", "out0"): reflection_fiber,
        }
    )
