""" SAX Photonic Integrated Circuit models """

import jax.numpy as jnp
from ..utils import zero
from ..typing import Dict, ModelDict, ComplexFloat


#########################
## Waveguides ##
#########################


def wg_transmission(params: Dict[str, float]) -> ComplexFloat:
    neff = params["neff"]
    dwl = params["wl"] - params["wl0"]
    dneff_dwl = (params["ng"] - params["neff"]) / params["wl0"]
    neff = neff - dwl * dneff_dwl
    phase = jnp.exp(
        jnp.log(2.0 * jnp.pi * neff * params["length"]) - jnp.log(params["wl"])
    )
    return 10 ** (-params["loss"] * params["length"] / 20) * jnp.exp(1j * phase)


wg = {
    ("in", "out"): wg_transmission,
    ("out", "in"): wg_transmission,
    "params": {
        "length": 25e-6,
        "wl": 1.55e-6,
        "wl0": 1.55e-6,
        "neff": 2.34,
        "ng": 3.4,
        "loss": 0.0,
    },
}
""" waveguide model """

#########################
## Directional coupler ##
#########################


def dc_coupling(params: Dict[str, float]) -> ComplexFloat:
    return 1j * params["coupling"] ** 0.5


def dc_transmission(params: Dict[str, float]) -> ComplexFloat:
    return (1 - params["coupling"]) ** 0.5


dc = {
    ("p0", "p1"): dc_transmission,
    ("p1", "p0"): dc_transmission,
    ("p2", "p3"): dc_transmission,
    ("p3", "p2"): dc_transmission,
    ("p0", "p2"): dc_coupling,
    ("p2", "p0"): dc_coupling,
    ("p1", "p3"): dc_coupling,
    ("p3", "p1"): dc_coupling,
    "params": {"coupling": 0.5},
}
""" directional coupler model """
