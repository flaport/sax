""" Tests for sax.core """

import sax
import jax
import jax.numpy as jnp
from pytest import approx

from .fixtures import dc, rdc, mzi, wg


def test_model(dc):
    assert dc.funcs["p0", "p1"] == dc.funcs["p1", "p0"]
    assert dc.funcs["p2", "p3"] == dc.funcs["p3", "p2"]
    assert dc.funcs["p0", "p2"] == dc.funcs["p2", "p0"]
    assert dc.funcs["p1", "p3"] == dc.funcs["p3", "p1"]
    assert tuple(dc.params.keys()) == ("coupling",)


def test_reciprocal_model(dc, rdc):
    assert rdc.funcs["p0", "p1"] == rdc.funcs["p1", "p0"]
    assert rdc.funcs["p2", "p3"] == rdc.funcs["p3", "p2"]
    assert rdc.funcs["p0", "p2"] == rdc.funcs["p2", "p0"]
    assert rdc.funcs["p1", "p3"] == rdc.funcs["p3", "p1"]
    assert dc.funcs["p0", "p1"] == rdc.funcs["p1", "p0"]
    assert dc.funcs["p2", "p3"] == rdc.funcs["p3", "p2"]
    assert dc.funcs["p0", "p2"] == rdc.funcs["p2", "p0"]
    assert dc.funcs["p1", "p3"] == rdc.funcs["p3", "p1"]
    assert tuple(rdc.params.keys()) == ("coupling",)


def test_circuit_mzi(mzi):
    assert tuple(sorted(mzi.params.keys())) == ("btm", "lft", "rgt", "top")
    mzi.params["lft"]["coupling"] = 0.5
    mzi.params["rgt"]["coupling"] = 0.5
    mzi.params["top"]["length"] = 50e-6
    mzi.params["btm"]["length"] = 50e-6
    assert abs(mzi.funcs["in0", "out0"](mzi.params)) == approx(0.0)
    mzi.params["btm"]["length"] = 25e-6
    assert abs(mzi.funcs["in0", "out0"](mzi.params)) == approx(0.7248724)
    f = jax.jit(mzi.funcs["in0", "out0"])
    assert abs(f(mzi.params)) == approx(0.7248724)
    assert abs(f(mzi.params)) == approx(0.7248724)
    g = jax.grad(lambda params: jnp.abs(mzi.funcs["in0", "out0"](params)) ** 2)
    assert g(mzi.params)["btm"]["length"] == approx(4736649.0)


def test_circuit_mzi_mzi(wg, mzi):
    mzi.params["lft"]["coupling"] = 0.5
    mzi.params["rgt"]["coupling"] = 0.5
    mzi.params["top"]["length"] = 50e-6
    mzi.params["btm"]["length"] = 50e-6
    mzi_mzi = sax.circuit(
        models={
            "lft": mzi,
            "top": wg,
            "rgt": mzi,
            "btm": wg,
        },
        connections={
            "lft:out1": "top:in",
            "lft:out0": "btm:in",
            "top:out": "rgt:in1",
            "btm:out": "rgt:in0",
        },
        ports={
            "lft:in1": "in1",
            "lft:in0": "in0",
            "rgt:out1": "out1",
            "rgt:out0": "out0",
        },
    )
    assert tuple(sorted(mzi_mzi.params.keys())) == ("btm", "lft", "rgt", "top")
    assert abs(mzi_mzi.funcs["in0", "out0"](mzi_mzi.params)) == approx(1.0)
    mzi_mzi.params["lft"]["btm"]["length"] = 25e-6
    mzi_mzi.params["rgt"]["btm"]["length"] = 25e-6
    mzi_mzi.params["btm"]["length"] = 25e-6
    assert abs(mzi_mzi.funcs["in0", "out0"](mzi_mzi.params)) == approx(1.0)
    f = jax.jit(mzi_mzi.funcs["in0", "out0"])
    assert abs(f(mzi_mzi.params)) == approx(1.0)
    assert abs(f(mzi_mzi.params)) == approx(1.0)
    g = jax.grad(lambda params: jnp.abs(mzi_mzi.funcs["in0", "out0"](params)) ** 2)
    assert g(mzi_mzi.params)["btm"]["length"] == approx(-0.4917065)
