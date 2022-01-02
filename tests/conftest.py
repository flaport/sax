""" fixtures for SAX tests """

from pytest import fixture

import sax


@fixture
def dc():
    return sax.models.pic.coupler()


@fixture
def wg():
    return sax.models.pic.straight()


@fixture
def mzi(dc, wg):
    mzi = sax.circuit(
        instances={
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
