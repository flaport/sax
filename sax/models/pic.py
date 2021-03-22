""" SAX Photonic Integrated Circuit models """

from __future__ import annotations

from ..core import model
from ..funcs.pic import wg_transmission, dc_transmission, dc_coupling


wg = model(
    funcs={
        ("in", "out"): wg_transmission,
        ("out", "in"): wg_transmission,
    },
    params={
        "length": 25e-6,
        "wl": 1.55e-6,
        "wl0": 1.55e-6,
        "neff": 2.34,
        "ng": 3.4,
        "loss": 0.0,
    },
)
""" waveguide model """

dc = model(
    funcs={
        ("p0", "p1"): dc_transmission,
        ("p1", "p0"): dc_transmission,
        ("p2", "p3"): dc_transmission,
        ("p3", "p2"): dc_transmission,
        ("p0", "p2"): dc_coupling,
        ("p2", "p0"): dc_coupling,
        ("p1", "p3"): dc_coupling,
        ("p3", "p1"): dc_coupling,
    },
    params={"coupling": 0.5},
)
""" directional coupler model """
