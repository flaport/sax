""" SAX thin-film models """

from __future__ import annotations

import jax.numpy as jnp
from sax._typing import Float, SDict


def fresnel_mirror_ij(ni: Float = 1.0, nj: Float = 1.0) -> SDict:
    """Model a (fresnel) interface between twoo refractive indices

    Args:
        ni: refractive index of the initial medium
        nf: refractive index of the final
    """
    r_fresnel_ij: Float = (ni - nj) / (ni + nj)  # i->j reflection
    t_fresnel_ij: Float = 2.0 * ni / (ni + nj)  # i->j transmission
    r_fresnel_ji = -r_fresnel_ij  # j -> i reflection
    t_fresnel_ji = (1.0 - r_fresnel_ij ** 2) / t_fresnel_ij  # j -> i transmission
    sdict = {
        ("in", "in"): r_fresnel_ij,
        ("in", "out"): t_fresnel_ij,
        ("out", "in"): t_fresnel_ji,
        ("out", "out"): r_fresnel_ji,
    }
    return sdict


def propagation_i(wl: Float = 0.532, ni: Float = 1.0, di: Float = 0.5) -> SDict:
    """Model the phase shift acquired as a wave propagates through medium A

    Args:
        ni: refractive index of medium (at wavelength wl)
        di: [μm] thickness of layer
        wl: [μm] wavelength
    """
    prop_i = jnp.exp(1j * 2 * jnp.pi * ni * di / wl)
    sdict = {
        ("in", "out"): prop_i,
        ("out", "in"): prop_i,
    }
    return sdict


def mirror(t_amp: Float = 0.5 ** 0.5, t_ang: Float = 0.0) -> SDict:
    r_complex_val = _r_complex(t_amp, t_ang)
    t_complex_val = _t_complex(t_amp, t_ang)
    sdict = {
        ("in", "in"): r_complex_val,
        ("in", "out"): t_complex_val,
        ("out", "in"): t_complex_val,  # (1 - r_complex_val**2)/t_complex_val, # t_ji
        ("out", "out"): r_complex_val,  # -r_complex_val, # r_ji
    }
    return sdict


def _t_complex(t_amp, t_ang):
    return t_amp * jnp.exp(-1j * t_ang)


def _r_complex(t_amp, t_ang):
    r_amp = jnp.sqrt((1.0 - t_amp ** 2))
    r_ang = t_ang - jnp.pi / 2
    return r_amp * jnp.exp(-1j * r_ang)
