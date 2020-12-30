import jax.numpy as jnp
from .core import model
from .constants import pi


@model(
    ports=("in", "out"),
    default_params={
        "length": 25e-6,
        "wl0": 1.55e-6,
        "neff": 2.34,
        "ng": 3.4,
        "loss": 0.0,
    },
    default_env={"wl": 1.55e-6},
    reciprocal=True,
)
def model_waveguide(params, env, i, j):
    if i == j:
        return 0.0
    neff = params["neff"]
    dwl = env["wl"] - params["wl0"]
    dneff_dwl = (params["ng"] - params["neff"]) / params["wl0"]
    neff = neff - dwl * dneff_dwl
    phase = jnp.exp(jnp.log(2 * pi * neff * params["length"]) - jnp.log(env["wl"]))
    return 10 ** (-params["loss"] * params["length"] / 20) * jnp.exp(1j * phase)


@model(
    ports=["p0", "p1", "p2", "p3"], default_params={"coupling": 0.5}, reciprocal=True
)
def model_directional_coupler(params, env, i, j):
    if i == j or i == 1 and j == 2 or i == 0 and j == 3:
        return 0
    elif i == 0 and j == 1 or i == 2 and j == 3:
        return (1 - params["coupling"]) ** 0.5
    elif i == 0 and j == 2 or i == 1 and j == 3:
        return 1j * params["coupling"] ** 0.5
