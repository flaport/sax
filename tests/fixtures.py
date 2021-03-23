""" fixtures for SAX tests """

from pytest import fixture

import sax
from sax.funcs.pic import dc_transmission, dc_coupling, wg_transmission


@fixture
def dc():
    dc = sax.model(
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
        params={"coupling": 0.3},
    )
    return dc


@fixture
def rdc():
    dc = sax.model(
        funcs={
            ("p0", "p1"): dc_transmission,
            ("p2", "p3"): dc_transmission,
            ("p0", "p2"): dc_coupling,
            ("p1", "p3"): dc_coupling,
        },
        params={"coupling": 0.3},
        reciprocal=True,
    )
    return dc


@fixture
def wg():
    wg = sax.model(
        funcs={
            ("in", "out"): wg_transmission,
        },
        params={
            "length": 25e-6,
            "wl": 1.55e-6,
            "wl0": 1.55e-6,
            "neff": 2.34,
            "ng": 3.4,
            "loss": 0.0,
        },
        reciprocal=True,
    )
    return wg


@fixture
def mzi(dc, wg):
    mzi = sax.circuit(
        models={
            "lft": dc,
            "top": wg,
            "rgt": dc,
            "btm": wg,
        },
        connections={
            "lft:p2": "top:in",
            "lft:p1": "btm:in",
            "top:out": "rgt:p3",
            "btm:out": "rgt:p0",
        },
        ports={
            "lft:p3": "in1",
            "lft:p0": "in0",
            "rgt:p2": "out1",
            "rgt:p1": "out0",
        },
    )
    return mzi
