""" SAX thin-film models """

from __future__ import annotations

import jax.numpy as jnp
from ..core import model
from ..funcs.thinfilm import (
    r_fresnel_ij,
    t_fresnel_ij,
    t_fresnel_ji,
    r_fresnel_ji,
    prop_i,
    r_complex,
    t_complex,
    t_complex,
    r_complex,
)


fresnel_mirror_ij = model(
    funcs={
        ("in", "in"): r_fresnel_ij,
        ("in", "out"): t_fresnel_ij,
        ("out", "in"): t_fresnel_ji,
        ("out", "out"): r_fresnel_ji,
    },
    params={
        "ni": 1.0,
        "nj": 1.0,
    },
)
""" fresnel interface """


propagation_i = model(
    funcs={
        ("in", "out"): prop_i,
        ("out", "in"): prop_i,
    },
    params={
        "ni": 1.0,
        "di": 500.0,
        "wl": 532.0,
    },
)
""" propagation phase """


mirror = model(
    funcs={
        ("in", "in"): r_complex,
        ("in", "out"): t_complex,
        ("out", "in"): t_complex,
        ("out", "out"): r_complex,
    },
    params={
        "t_amp": jnp.sqrt(0.5),
        "t_ang": 0.0,
    },
)
""" fresnel mirror """
