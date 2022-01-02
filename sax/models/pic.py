""" SAX Photonic Integrated Circuit models """

from __future__ import annotations

import jax.numpy as jnp

from sax.utils import reciprocal
from sax._typing import SDict, Float


def straight(
    *,
    wl: Float = 1.55,
    wl0: Float = 1.55,
    neff: Float = 2.34,
    ng: Float = 3.4,
    length: Float = 10.0,
    loss: Float = 0.0
) -> SDict:
    """a simple straight waveguide model

    Args:
        wl: wavelength
        neff: waveguide effective index
        ng: waveguide group index (used for linear neff dispersion)
        wl0: center wavelength at which neff is defined
        length: [m] wavelength length
        loss: [dB/m] waveguide loss
    """
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * neff * length / wl
    transmission = 10 ** (-loss * length / 20) * jnp.exp(1j * phase)
    sdict = reciprocal(
        {
            ("in0", "out0"): transmission,
        }
    )
    return sdict


def coupler(*, coupling: Float = 0.5) -> SDict:
    kappa = coupling ** 0.5
    tau = (1 - coupling) ** 0.5
    sdict = reciprocal(
        {
            ("in0", "out0"): tau,
            ("in0", "out1"): 1j * kappa,
            ("in1", "out0"): 1j * kappa,
            ("in1", "out1"): tau,
        }
    )
    return sdict


def dc_transmission(**kwargs):
    pass


def dc_coupling(**kwargs):
    pass


def wg_transmission(**kwargs):
    pass


if __name__ == "__main__":
    c = coupler(coupling=0.5)
