import jax.numpy as jnp
from .utils import zero
from .core import modelgenerator


def model_waveguide_transmission(params):
    neff = params["neff"]
    dwl = params["wl"] - params["wl0"]
    dneff_dwl = (params["ng"] - params["neff"]) / params["wl0"]
    neff = neff - dwl * dneff_dwl
    phase = jnp.exp(
        jnp.log(2 * jnp.pi * neff * params["length"]) - jnp.log(params["wl"])
    )
    return 10 ** (-params["loss"] * params["length"] / 20) * jnp.exp(1j * phase)


@modelgenerator(
    ports=("in", "out"),
    default_params={
        "length": 25e-6,
        "wl": 1.55e-6,
        "wl0": 1.55e-6,
        "neff": 2.34,
        "ng": 3.4,
        "loss": 0.0,
    },
    reciprocal=True,
)
def model_waveguide(i, j):
    if i == j:
        return
    return model_waveguide_transmission


def model_directional_coupler_coupling(params):
    return 1j * params["coupling"] ** 0.5


def model_directional_coupler_transmission(params):
    return (1 - params["coupling"]) ** 0.5


@modelgenerator(
    ports=("p0", "p1", "p2", "p3"),
    default_params={"coupling": 0.5},
    reciprocal=True,
)
def model_directional_coupler(i, j):
    if i == 0 and j == 1 or i == 2 and j == 3:
        return model_directional_coupler_transmission
    elif i == 0 and j == 2 or i == 1 and j == 3:
        return model_directional_coupler_coupling
