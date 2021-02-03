""" SAX Photonic Integrated Circuit models """

import jax.numpy as jnp
from ..utils import zero
from ..core import modelgenerator
from ..typing import Dict, ModelDict, ComplexFloat


#########################
## Waveguides ##
#########################

def model_waveguide_transmission(params: Dict[str, float]) -> ComplexFloat:
    neff = params["neff"]
    dwl = params["wl"] - params["wl0"]
    dneff_dwl = (params["ng"] - params["neff"]) / params["wl0"]
    neff = neff - dwl * dneff_dwl
    phase = jnp.exp(
        jnp.log(2 * jnp.pi * neff * params["length"]) - jnp.log(params["wl"])
    )
    return 10 ** (-params["loss"] * params["length"] / 20) * jnp.exp(1j * phase)

waveguide: ModelDict = {
    ("in", "out"): model_waveguide_transmission,
    ("out", "in"): model_waveguide_transmission,
    "default_params": {
        "length": 25e-6,
        "wl": 1.55e-6,
        "wl0": 1.55e-6,
        "neff": 2.34,
        "ng": 3.4,
        "loss": 0.0,
    },
}

#########################
## Directional coupler ##
#########################

def model_directional_coupler_coupling(params: Dict[str, float]) -> ComplexFloat:
    return 1j * params["coupling"] ** 0.5

def model_directional_coupler_transmission(params: Dict[str, float]) -> ComplexFloat:
    return (1 - params["coupling"]) ** 0.5

directional_coupler: ModelDict = {
    ("p0", "p1"): model_directional_coupler_transmission,
    ("p1", "p0"): model_directional_coupler_transmission,
    ("p2", "p3"): model_directional_coupler_transmission,
    ("p3", "p2"): model_directional_coupler_transmission,
    ("p0", "p2"): model_directional_coupler_coupling,
    ("p2", "p0"): model_directional_coupler_coupling,
    ("p1", "p3"): model_directional_coupler_coupling,
    ("p3", "p1"): model_directional_coupler_coupling,
    "default_params": {"coupling": 0.5},
}