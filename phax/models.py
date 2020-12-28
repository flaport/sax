import jax.numpy as jnp
from .core import model
from .constants import pi


@model(num_ports=2, reciprocal=True)
def model_waveguide(
    i, j, wl=1.55e-6, length=1e-5, loss=0.0, neff0=2.34, ng=3.4, wl0=1.55e-6
):
    if i == j:
        return 0.0
    neff = neff0 - (wl - wl0) * (ng - neff0) / wl0
    phase = jnp.exp(jnp.log(2 * pi * neff * length) - jnp.log(wl))
    return 10 ** (-loss * length / 20) * jnp.exp(1j * phase)


@model(num_ports=4, reciprocal=True)
def model_directional_coupler(i, j, wl=1.55e-6, coupling=0.5):
    if i == j or i == 1 and j == 2 or i == 0 and j == 3:
        return 0
    elif i == 0 and j == 1 or i == 2 and j == 3:
        return (1 - coupling) ** 0.5
    elif i == 0 and j == 2 or i == 1 and j == 3:
        return 1j * coupling ** 0.5
