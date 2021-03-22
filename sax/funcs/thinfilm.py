""" SAX thin-film model functions """

from __future__ import annotations

import jax.numpy as jnp
from typing import Dict
from .._typing import ComplexFloat, ComplexFloat
from ..constants import pi

__all__ = [
    "prop_i",
    "r_complex",
    "r_fresnel_ij",
    "r_fresnel_ji",
    "t_complex",
    "t_fresnel_ij",
    "t_fresnel_ji",
]


def prop_i(params: Dict[str, ComplexFloat]) -> ComplexFloat:
    """Phase shift acquired as a wave propagates through medium i

    Args:
        wl : wavelength
        ni : refractive index of medium (at wavelength wl)
        di : thickness of layer (same arb. unit as wl)
    """
    return jnp.exp(1j * 2 * pi * params["ni"] / params["wl"] * params["di"])


def r_complex(params: Dict[str, ComplexFloat]) -> ComplexFloat:
    """Reflection coefficient

    Args:
        t_amp: transmission amplitude
        t_ang: transmission phase
    """
    r_amp = jnp.sqrt((1.0 - params["t_amp"] ** 2))
    r_ang = params["t_ang"] - pi / 2
    return r_amp * jnp.exp(-1j * r_ang)


def r_fresnel_ij(params: Dict[str, ComplexFloat]) -> ComplexFloat:
    """Normal incidence amplitude reflection from Fresnel's equations

    Args:
        ni : refractive index of the initial medium
        nj : refractive index of the final medium
    """
    return (params["ni"] - params["nj"]) / (params["ni"] + params["nj"])


def r_fresnel_ji(params: Dict[str, ComplexFloat]) -> ComplexFloat:
    """Normal incidence amplitude reflection from Fresnel's equations

    Args:
        ni : refractive index of the initial medium
        nj : refractive index of the final medium
    """
    return -1 * r_fresnel_ij(params)


def t_complex(params: Dict[str, ComplexFloat]) -> ComplexFloat:
    """Transmission coefficient (design parameter)

    Args:
        t_amp: transmission amplitude
        t_ang: transmission phase
    """
    return params["t_amp"] * jnp.exp(-1j * params["t_ang"])


def t_fresnel_ij(params: Dict[str, ComplexFloat]) -> ComplexFloat:
    """Normal incidence amplitude transmission from Fresnel's equations

    Args:
        ni : refractive index of the initial medium
        nj : refractive index of the final medium
    """
    return 2 * params["ni"] / (params["ni"] + params["nj"])


def t_fresnel_ji(params: Dict[str, ComplexFloat]) -> ComplexFloat:
    """Normal incidence amplitude transmission from Fresnel's equations

    Args:
        ni : refractive index of the initial medium
        nj : refractive index of the final medium
    """
    return (1 - r_fresnel_ij(params) ** 2) / t_fresnel_ij(params)
