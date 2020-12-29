import jax.numpy as jnp
from .core import model
from .constants import pi


@model(ports=("in", "out"), reciprocal=True)
def model_waveguide(params, env, i, j):
    if i == j:
        return 0.0
    neff = params["neff0"] - (env["wl"] - params["wl0"]) * (params["ng"] - params["neff0"]) / params["wl0"]
    phase = jnp.exp(jnp.log(2 * pi * neff * params["length"]) - jnp.log(env["wl"]))
    return 10 ** (-params["loss"] * params["length"] / 20) * jnp.exp(1j * phase)


@model(ports=["p0", "p1", "p2", "p3"], reciprocal=True)
def model_directional_coupler(params, env, i, j):
    if i == j or i == 1 and j == 2 or i == 0 and j == 3:
        return 0
    elif i == 0 and j == 1 or i == 2 and j == 3:
        return (1 - params["coupling"]) ** 0.5
    elif i == 0 and j == 2 or i == 1 and j == 3:
        return 1j * params["coupling"] ** 0.5
