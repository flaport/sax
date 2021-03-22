""" SAX Photonic Integrated Circuit model functions """

from __future__ import annotations

import jax.numpy as jnp
from typing import Dict
from .._typing import ComplexFloat, ComplexFloat
from ..constants import pi

__all__ = ["dc_coupling", "dc_transmission", "wg_transmission"]


def dc_coupling(params: Dict[str, ComplexFloat]) -> ComplexFloat:
    """Directional coupler coupling

    Args:
        coupling: power coupling of the coupler
    """
    return 1j * params["coupling"] ** 0.5


def dc_transmission(params: Dict[str, ComplexFloat]) -> ComplexFloat:
    """Directional coupler transmission

    Args:
        coupling: power coupling of the coupler (=1-transmission)
    """
    return (1 - params["coupling"]) ** 0.5


def wg_transmission(params: Dict[str, ComplexFloat]) -> ComplexFloat:
    """Waveguide transmission

    Args:
        wl: wavelength
        neff: waveguide effective index
        ng: waveguide group index (used for linear neff dispersion)
        wl0: center wavelength at which neff is defined
        length: [m] wavelength length
        loss: [dB/m] waveguide loss
    """
    neff = params["neff"]
    dwl = params["wl"] - params["wl0"]
    dneff_dwl = (params["ng"] - params["neff"]) / params["wl0"]
    neff = neff - dwl * dneff_dwl
    phase = jnp.exp(jnp.log(2.0 * pi * neff * params["length"]) - jnp.log(params["wl"]))
    return 10 ** (-params["loss"] * params["length"] / 20) * jnp.exp(1j * phase)
