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
# default_exp backends.default
# -

# # Backend - default
#
# > Default SAX Backend

# hide
import os, sys; sys.stderr = open(os.devnull, "w")

# +
# export
from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Dict

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    def jit(func, *args, **kwargs):
        warnings.warn("[NO JAX] skipping jit! Please install JAX!")
        return func
    
    jax = SimpleNamespace(jit=jit)
    JAX_AVAILABLE = False
    
from sax.typing_ import SDict, SType, SDense, sdict


# -

# ## Citation
# The default SAX backend is based on the following paper:
#
# > Filipsson, Gunnar. "*A new general computer algorithm for S-matrix calculation of interconnected multiports.*" 11th European Microwave Conference. IEEE, 1981.

# ## Circuit Evaluation

# +
# export

def evaluate_circuit(
    instances: Dict[str, SType],
    connections: Dict[str, str],
    ports: Dict[str, str],
) -> SDict:
    """evaluate a circuit for the given sdicts."""
    
    # it's actually easier working w reverse:
    reversed_ports = {v: k for k, v in ports.items()}

    block_diag = {}
    for name, S in instances.items():
        block_diag.update(
            {(f"{name},{p1}", f"{name},{p2}"): v for (p1, p2), v in sdict(S).items()}
        )

    sorted_connections = sorted(connections.items(), key=_connections_sort_key)
    all_connected_instances = {k: {k} for k in instances}

    for k, l in sorted_connections:
        name1, _ = k.split(",")
        name2, _ = l.split(",")

        connected_instances = (
            all_connected_instances[name1] | all_connected_instances[name2]
        )
        for name in connected_instances:
            all_connected_instances[name] = connected_instances

        current_ports = tuple(
            p
            for instance in connected_instances
            for p in set([p for p, _ in block_diag] + [p for _, p in block_diag])
            if p.startswith(f"{instance},")
        )

        block_diag.update(_interconnect_ports(block_diag, current_ports, k, l))

        for i, j in list(block_diag.keys()):
            is_connected = i == k or i == l or j == k or j == l
            is_in_output_ports = i in reversed_ports and j in reversed_ports
            if is_connected and not is_in_output_ports:
                del block_diag[i, j]  # we're no longer interested in these port combinations

    circuit_sdict: SDict = {
        (reversed_ports[i], reversed_ports[j]): v
        for (i, j), v in block_diag.items()
        if i in reversed_ports and j in reversed_ports
    }
    return circuit_sdict


def _connections_sort_key(connection):
    """sort key for sorting a connection dictionary """
    part1, part2 = connection
    name1, _ = part1.split(",")
    name2, _ = part2.split(",")
    return (min(name1, name2), max(name1, name2))


def _interconnect_ports(block_diag, current_ports, k, l):
    """interconnect two ports in a given model

    > Note: the interconnect algorithm is based on equation 6 of 'Filipsson, Gunnar. 
      "A new general computer algorithm for S-matrix calculation of interconnected 
      multiports." 11th European Microwave Conference. IEEE, 1981.'
    """
    current_block_diag = {}
    for i in current_ports:
        for j in current_ports:
            vij = _calculate_interconnected_value(
                vij=block_diag.get((i, j), 0.0),
                vik=block_diag.get((i, k), 0.0),
                vil=block_diag.get((i, l), 0.0),
                vkj=block_diag.get((k, j), 0.0),
                vkk=block_diag.get((k, k), 0.0),
                vkl=block_diag.get((k, l), 0.0),
                vlj=block_diag.get((l, j), 0.0),
                vlk=block_diag.get((l, k), 0.0),
                vll=block_diag.get((l, l), 0.0),
            )
            current_block_diag[i, j] = vij
    return current_block_diag


@jax.jit
def _calculate_interconnected_value(vij, vik, vil, vkj, vkk, vkl, vlj, vlk, vll):
    """Calculate an interconnected S-parameter value

    Note:
        The interconnect algorithm is based on equation 6 in the paper below::

          Filipsson, Gunnar. "A new general computer algorithm for S-matrix calculation
          of interconnected multiports." 11th European Microwave Conference. IEEE, 1981.
    """
    result = vij + (
        vkj * vil * (1 - vlk)
        + vlj * vik * (1 - vkl)
        + vkj * vll * vik
        + vlj * vkk * vil
    ) / ((1 - vkl) * (1 - vlk) - vkk * vll)
    return result


# -

# ## Example

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
# -

# ## Algorithm Walkthrough
#
# > Note: This algorithm gets pretty slow for large circuits. I'd be [very interested in any improvements](#Algorithm-Improvements) that can be made here, especially because - as opposed to the currently faster [KLU backend](./07b_backends_klu.ipynb) - the algorithm discussed here is jittable, differentiable and can be used on GPUs.

# Let's walk through all the steps of this algorithm. We'll do this for a simple MZI circuit, given by two directional couplers characterised by `dc_sdense` with a phase shifting waveguide in between `wg_sdict`:

instances={
    "dc1": dc_sdense,
    "wg": wg_sdict,
    "dc2": dc_sdense,
}
connections={
    "dc1,out0": "wg,in0",
    "wg,out0": "dc2,in0",
    "dc1,out1": "dc2,in1",
}
ports={
    "in0": "dc1,in0",
    "in1": "dc1,in1",
    "out0": "dc2,out0",
    "out1": "dc2,out1",
}

# as a first step, we construct the `reversed_ports`, it's actually easier to work with `reversed_ports` (we chose the opposite convention in the netlist definition to adhere to the GDSFactory netlist convention):

reversed_ports = {v: k for k, v in ports.items()}

# The first real step of the algorithm is to create the 'block diagonal sdict`:

block_diag = {}
for name, S in instances.items():
    block_diag.update(
        {(f"{name},{p1}", f"{name},{p2}"): v for (p1, p2), v in sdict(S).items()}
    )


# we can optionally filter out zeros from the resulting block_diag representation. Just note that this will make the resuling function unjittable (the resulting 'shape' (i.e. keys) of the dictionary would depend on the data itself, which is not allowed in JAX jit). We're doing it here to avoid printing zeros but **internally this is not done by default**.

block_diag = {k: v for k, v in block_diag.items() if jnp.abs(v) > 1e-10}
print(len(block_diag))
block_diag

# next, we sort the connections such that similar components are grouped together:

sorted_connections = sorted(connections.items(), key=_connections_sort_key)
sorted_connections

# Now we iterate over the sorted connections and connect components as they come in. Connected components take over the name of the first component in the connection, but we keep a set of components belonging to that key in `all_connected_instances`.
#
# This is how this `all_connected_instances` dictionary looks initially.

all_connected_instances = {k: {k} for k in instances}
all_connected_instances

# Normally we would loop over every connection in `sorted_connections` now, but let's just go through it once at first:

# for k, l in sorted_connections:
k, l = sorted_connections[0]
k, l

# `k` and `l` are the S-matrix indices we're trying to connect. Note that in our sparse `SDict` notation these S-matrix indices are in fact equivalent with the port names `('dc1,out1', 'dc2,in1')`!

# first we split the connection string into an instance name and a port name (we don't use the port name yet):

name1, _ = k.split(",")
name2, _ = l.split(",")

# We then obtain the new set of connected instances.

connected_instances = all_connected_instances[name1] | all_connected_instances[name2]
connected_instances

# We then iterate over each of the components in this set and make sure each of the component names in that set maps to that set (yes, I know... confusing). We do this to be able to keep track with which components each of the components in the circuit is currently already connected to.

# +
for name in connected_instances:
    all_connected_instances[name] = connected_instances
    
all_connected_instances
# -

# now we need to obtain all the ports of the currently connected instances.

# +
current_ports = tuple(
    p
    for instance in connected_instances
    for p in set([p for p, _ in block_diag] + [p for _, p in block_diag])
    if p.startswith(f"{instance},")
)

current_ports


# -

# Now the [Gunnar Algorithm](#citation) is used. Given a (block-diagonal) 'S-matrix' `block_diag` and a 'connection matrix' `current_ports` we can interconnect port `k` and `l` as follows:
#
# > Note: some creative freedom is used here. In SAX, the matrices we're talking about are in fact represented by a sparse dictionary (an `SDict`), i.e. similar to a COO sparse matrix for which the indices are the port names.

# +
def _interconnect_ports(block_diag, current_ports, k, l):
    current_block_diag = {}
    for i in current_ports:
        for j in current_ports:
            vij = _calculate_interconnected_value(
                vij=block_diag.get((i, j), 0.0),
                vik=block_diag.get((i, k), 0.0),
                vil=block_diag.get((i, l), 0.0),
                vkj=block_diag.get((k, j), 0.0),
                vkk=block_diag.get((k, k), 0.0),
                vkl=block_diag.get((k, l), 0.0),
                vlj=block_diag.get((l, j), 0.0),
                vlk=block_diag.get((l, k), 0.0),
                vll=block_diag.get((l, l), 0.0),
            )
            current_block_diag[i, j] = vij
    return current_block_diag

@jax.jit
def _calculate_interconnected_value(vij, vik, vil, vkj, vkk, vkl, vlj, vlk, vll):
    result = vij + (
        vkj * vil * (1 - vlk)
        + vlj * vik * (1 - vkl)
        + vkj * vll * vik
        + vlj * vkk * vil
    ) / ((1 - vkl) * (1 - vlk) - vkk * vll)
    return result

block_diag.update(_interconnect_ports(block_diag, current_ports, k, l))
# -

# Just as before, we're filtering the zeros from the sparse representation (remember, internally this is **not done by default**).

block_diag = {k: v for k, v in block_diag.items() if jnp.abs(v) > 1e-10}
print(len(block_diag))
block_diag

# This is the resulting block-diagonal matrix after interconnecting two ports (i.e. basically saying that those two ports are the same port). Because these ports are now connected we should actually remove them from the S-matrix representation (they are integrated into the S-parameters of the other connections):

# +
for i, j in list(block_diag.keys()):
    is_connected = i == k or i == l or j == k or j == l
    is_in_output_ports = i in reversed_ports and j in reversed_ports
    if is_connected and not is_in_output_ports:
        del block_diag[i, j]  # we're no longer interested in these port combinations
        
print(len(block_diag))
block_diag
# -

# Note that this deletion of values **does NOT** make this operation un-jittable. The deletion depends on the ports of the dictionary (i.e. on the dictionary 'shape'), not on the values.

# We now basically have to do those steps again for all other connections:

#for k, l in sorted_connections: 
for k, l in sorted_connections[1:]: # we just did the first iteration of this loop above...
    name1, _ = k.split(",")
    name2, _ = l.split(",")
    connected_instances = all_connected_instances[name1] | all_connected_instances[name2]
    for name in connected_instances:
        all_connected_instances[name] = connected_instances
    current_ports = tuple(
        p
        for instance in connected_instances
        for p in set([p for p, _ in block_diag] + [p for _, p in block_diag])
        if p.startswith(f"{instance},")
    )
    block_diag.update(_interconnect_ports(block_diag, current_ports, k, l))
    for i, j in list(block_diag.keys()):
        is_connected = i == k or i == l or j == k or j == l
        is_in_output_ports = i in reversed_ports and j in reversed_ports
        if is_connected and not is_in_output_ports:
            del block_diag[i, j]  # we're no longer interested in these port combinations

# This is the final MZI matrix we're getting:

block_diag

# All that's left is to rename these internal ports of the format `{instance},{port}` into output ports of the resulting circuit:

circuit_sdict: SDict = {
    (reversed_ports[i], reversed_ports[j]): v
    for (i, j), v in block_diag.items()
    if i in reversed_ports and j in reversed_ports
}
circuit_sdict

# And that's it. We evaluated the `SDict` of the full circuit.

# ## Algorithm Improvements

# This algorithm is 
#
# * pretty fast for small circuits 🙂
# * jittable 🙂
# * differentiable 🙂
# * GPU-compatible 🙂
#
# This algorithm is however:
#
# * **really slow** for large circuits 😥
# * **pretty slow** to jit the resulting circuit function 😥
# * **pretty slow** to differentiate the resulting circuit function 😥
#
# There are probably still plenty of improvements possible for this algorithm:
#
# * **¿** Network analysis (ft. NetworkX ?) to obtain which ports of the block diagonal representation are relevant to obtain the output connection **?**
# * **¿** Smarter ordering of connections to always have the minimum amount of ports in the intermediate block-diagonal representation **?**
# * **¿** Using `jax.lax.scan` in stead of python native for-loops in `_interconnect_ports` **?**
# * **¿** ... **?**
#
# Bottom line is... Do you know how to improve this algorithm or how to implement the above suggestions? Please open a Merge Request!
