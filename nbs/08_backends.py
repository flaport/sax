# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: sax
#     language: python
#     name: sax
# ---

# +
# default_exp backends.__init__
# -

# # Backend
#
# > SAX Backends

# +
# hide
import jax.numpy as jnp
from nbdev import show_doc
from sax.typing_ import SDense, SDict

import os, sys; sys.stderr = open(os.devnull, "w")

# +
# exporti
from __future__ import annotations

try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    
try:
    import klujax
    KLUJAX_AVAILABLE = True
except ImportError:
    KLUJAX_AVAILABLE = False

from sax.backends.default import evaluate_circuit
from sax.backends.klu import evaluate_circuit_klu
from sax.backends.additive import evaluate_circuit_additive
# -

# #### circuit_backends

# +
# exports

circuit_backends = {
    "default": evaluate_circuit,
    "klu": evaluate_circuit_klu,
    "additive": evaluate_circuit_additive,
}

if (not JAX_AVAILABLE) or (not KLUJAX_AVAILABLE):
    del circuit_backends["klu"]
# -

# SAX allows to easily interchange the backend of a circuit. A SAX backend needs to have the following signature:

# hide_input
from sax.backends.default import evaluate_circuit
show_doc(evaluate_circuit, doc_string=False)

# i.e. it takes a dictionary of instance names pointing to `SType`s (usually `SDict`s), a connection dictionary and an (output) ports dictionary. Internally it must construct the output `SType` (usually output `SDict`).

# > Example
#
# Let's create an MZI `SDict` using the default backend's `evaluate_circuit`:

# +
wg_sdict: SDict = {
    ("in0", "out0"): 0.5 + 0.86603j,
    ("out0", "in0"): 0.5 + 0.86603j,
}

τ, κ = 0.5 ** 0.5, 1j * 0.5 ** 0.5
dc_sdense: SDense = (
    jnp.array([[0, 0, τ, κ], 
               [0, 0, κ, τ], 
               [τ, κ, 0, 0], 
               [κ, τ, 0, 0]]),
    {"in0": 0, "in1": 1, "out0": 2, "out1": 3},
)

mzi_sdict: SDict = evaluate_circuit(
    instances={
        "dc1": dc_sdense,
        "wg": wg_sdict,
        "dc2": dc_sdense,
    },
    connections={
        "dc1,out0": "wg,in0",
        "wg,out0": "dc2,in0",
        "dc1,out1": "dc2,in1",
    },
    ports={
        "in0": "dc1,in0",
        "in1": "dc1,in1",
        "out0": "dc2,out0",
        "out1": "dc2,out1",
    }
)

mzi_sdict
