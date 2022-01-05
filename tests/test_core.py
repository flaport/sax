""" Tests for sax.core """

import sax
import jax.numpy as jnp

from sax.models import pic


def test_model_dc():
    dc = sax.models.pic.coupler()
    assert dc["in0", "out1"] == dc["in1", "out0"]


def test_circuit_mzi():
    mzi = sax.circuit(
        instances={
            "lft": pic.coupler,
            "top": pic.straight,
            "btm": pic.straight,
            "rgt": pic.coupler,
        },
        connections={
            "lft:out0": "btm:in0",
            "btm:out0": "rgt:in0",
            "lft:out1": "top:in0",
            "top:out0": "rgt:in1",
        },
        ports={
            "in0": "lft:in0",
            "in1": "lft:in1",
            "out0": "rgt:out0",
            "out1": "rgt:out1",
        },
    )
    params = sax.set_params(sax.get_params(mzi), wl=jnp.linspace(1.5, 1.6, 1))
    delta_length = 10
    params["top"]["length"] = 15.0 + delta_length
    params["btm"]["length"] = 15.0
    S = mzi(**params)
    return S


if __name__ == "__main__":
    s = test_circuit_mzi()
