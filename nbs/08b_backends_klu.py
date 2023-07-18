# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: octans
#     language: python
#     name: octans
# ---

# +
# default_exp backends.klu
# -

# # Backend - KLU
#
# > SAX KLU Backend

# +
# hide
import sax
import matplotlib.pyplot as plt
from fastcore.test import test_eq
from pytest import approx, raises
from nbdev import show_doc

import os, sys; sys.stderr = open(os.devnull, "w")

# +
# export
from __future__ import annotations

from typing import Dict

    
from sax.typing_ import SDense, SDict, SType, scoo
from sax.backends import evaluate_circuit

try:
    import klujax
except ImportError:
    klujax = None
    
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False
# -

# ## Citation
# The KLU backend is using `klujax`, which uses the [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) C++ libraries for sparse matrix evaluations to evaluate the circuit insanely fast on a CPU. The specific algorith being used in question is the KLU algorithm:
#
# > Ekanathan Palamadai Natariajan. "*KLU - A high performance sparse linear solver for circuit simulation problems.*"

# ## Theoretical Background

# The core of the KLU algorithm is supported by `klujax`, which internally uses the Suitesparse libraries to solve the sparse system `Ax = b`, in which A is a sparse matrix.

# Now it only comes down to shoehorn our circuit evaluation into a sparse linear system of equations $Ax=b$ where we need to solve for $x$ using `klujax`. 
# Consider the block diagonal matrix $S_{bd}$ of all components in the circuit acting on the fields $x_{in}$ at each of the individual ports of each of the component integrated in $S^{bd}$. The output fields $x^{out}$ at each of those ports is then given by:
#
# $$
# x^{out} = S_{bd} x^{in}
# $$

# However, $S_{bd}$ is not the S-matrix of the circuit as it does not encode any connectivity *between* the components. Connecting two component ports basically comes down to enforcing equality between the output fields at one port of a component with the input fields at another port of another (or maybe even the same) component. This equality can be enforced by creating an internal connection matrix, connecting all internal ports of the circuit:
#
# $$
# x^{in} = C_{int} x^{out}
# $$

# We can thus write the following combined equation:
#
# $$
# x^{in} = C_{int} S_{bd} x^{in}
# $$

# But this is not the complete story... Some component ports will *not* be *interconnected* with other ports: they will become the new *external ports* (or output ports) of the combined circuit. We can include those external ports into the above equation as follows:
#
# $$
# \begin{pmatrix} x^{in} \\ x^{out}_{ext} \end{pmatrix} = \begin{pmatrix} C_{int} & C_{ext} \\ C_{ext}^T & 0 \end{pmatrix} \begin{pmatrix} S_{bd} x^{in} \\ x_{ext}^{in} \end{pmatrix} 
# $$

# Note that $C_{ext}$ is obviously **not** a square matrix. Eliminating $x^{in}$ from the equation above finally yields:
#
# $$
# x^{out}_{ext} = C^T_{ext} S_{bd} (\mathbb{1} - C_{int}S_{bd})^{-1} C_{ext}x_{ext}^{in}
# $$

# We basically found a representation of the circuit S-matrix:
#
# $$
# S = C^T_{ext} S_{bd} (\mathbb{1} - C_{int}S_{bd})^{-1} C_{ext}
# $$

# Obviously, we won't want to calculate the inverse $(\mathbb{1} - C_{int}S_{bd})^{-1}$, which is the inverse of a very sparse matrix (a connection matrix only has a single 1 per line), which very often is not even sparse itself. In stead we'll use the `solve_klu` function:
#
# $$
# S = C^T_{ext} S_{bd} \texttt{solve}\_\texttt{klu}\left((\mathbb{1} - C_{int}S_{bd}), C_{ext}\right)
# $$

# Moreover, $C_{ext}^TS_{bd}$ is also a sparse matrix, therefore we'll also need a `mul_coo` routine:
#
# $$
# S = C^T_{ext} \texttt{mul}\_\texttt{coo}\left(S_{bd},~~\texttt{solve}\_\texttt{klu}\left((\mathbb{1} - C_{int}S_{bd}),~C_{ext}\right)\right)
# $$

# ## Sparse Helper Functions

# hide_input
if klujax is not None:
    show_doc(klujax.solve, doc_string=False, name="klujax.solve")

# `klujax.solve` solves the sparse system of equations `Ax=b` for `x`. Where `A` is represented by in [COO-format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)) as (`Ai`, `Aj`, `Ax`).
#
# > Example

# +
Ai = jnp.array([0, 1, 2, 3, 4])
Aj = jnp.array([1, 3, 4, 0, 2])
Ax = jnp.array([5, 6, 1, 1, 2])
b = jnp.array([5, 3, 2, 6, 1])

if klujax is not None:
    x = klujax.solve(Ai, Aj, Ax, b)
else:
    x = jnp.array([6.0, 1.0, 0.5, 0.5, 2.0])

x
# -

# This result is indeed correct:

if JAX_AVAILABLE:
    A = jnp.zeros((5, 5)).at[Ai, Aj].set(Ax)
    print(A)
    print(A@x)

# However, to use this function effectively, we probably need an extra dimension for `Ax`. Indeed, we would like to solve this equation for multiple wavelengths (or more general, for multiple circuit configurations) at once. For this we can use `jax.vmap` to expose `klujax.solve` to more dimensions for `Ax`:

# exports
solve_klu = None
if klujax is not None:
    solve_klu = jax.vmap(klujax.solve, (None, None, 0, None), 0)

# hide_input
show_doc(solve_klu, doc_string=False, name="solve_klu")

# Let's now redefine `Ax` and see what it gives:

# +
Ai = jnp.array([0, 1, 2, 3, 4])
Aj = jnp.array([1, 3, 4, 0, 2])
Ax = jnp.array([[5, 6, 1, 1, 2], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]])
b = jnp.array([5, 3, 2, 6, 1])
if klujax is not None:
    x = solve_klu(Ai, Aj, Ax, b)
else:
    x = jnp.array([
        [6.0, 1.0, 0.5, 0.5, 2.0],
        [3.0, 1.0, 1.0, 0.75, 0.66666667],
        [1.5, 5.0, 0.2, 1.5, 0.66666667],
    ])
    
x
# -

# This result is indeed correct:

if JAX_AVAILABLE:
    A = jnp.zeros((3, 5, 5)).at[:, Ai, Aj].set(Ax)
    jnp.einsum("ijk,ik->ij", A, x)

# Additionally, we need a way to multiply a sparse COO-matrix with a dense vector. This can be done with `klujax.coo_mul_vec`:

# +
# hide_input

if klujax is not None:
    show_doc(klujax.coo_mul_vec, doc_string=False, name="klujax.coo_mul_vec")
# -

# However, it's useful to allow a batch dimension, this time *both* in `Ax` and in `b`:

# +
# exporti

# @jax.jit  # TODO: make this available to autograd
# def mul_coo(Ai, Aj, Ax, b):
#     result = jnp.zeros_like(b).at[..., Ai, :].add(Ax[..., :, None] * b[..., Aj, :])
#     return result
# -

# exports
mul_coo = None 
if klujax is not None:
    mul_coo = jax.vmap(klujax.coo_mul_vec, (None, None, 0, 0), 0)

# hide_input
show_doc(mul_coo, doc_string=False, name="mul_coo")

# Let's confirm this does the right thing:

# +
if klujax is not None:
    result = mul_coo(Ai, Aj, Ax, x)
else:
    result = jnp.array([
        [5.0, 3.0, 2.0, 6.0, 1.0],
        [5.0, 3.0, 2.00000001, 6.0, 1.0],
        [5.0, 3.0, 2.00000001, 6.0, 1.0],
    ])
    
result


# -

# ## Circuit Evaluation

# export
def evaluate_circuit_klu(
    instances: Dict[str, SType],
    connections: Dict[str, str],
    ports: Dict[str, str],
):
    """evaluate a circuit using KLU for the given sdicts. """

    if klujax is None:
        raise ImportError(
            "Could not import 'klujax'. "
            "Please install it first before using backend method 'klu'"
        )

    assert solve_klu is not None
    assert mul_coo is not None

    connections = {**connections, **{v: k for k, v in connections.items()}}
    inverse_ports = {v: k for k, v in ports.items()}
    port_map = {k: i for i, k in enumerate(ports)}

    idx, Si, Sj, Sx, instance_ports = 0, [], [], [], {}
    batch_shape = ()
    for name, instance in instances.items():
        si, sj, sx, ports_map = scoo(instance)
        Si.append(si + idx)
        Sj.append(sj + idx)
        Sx.append(sx)
        if len(sx.shape[:-1]) > len(batch_shape):
            batch_shape = sx.shape[:-1]
        instance_ports.update({f"{name},{p}": i + idx for p, i in ports_map.items()})
        idx += len(ports_map)

    Si = jnp.concatenate(Si, -1)
    Sj = jnp.concatenate(Sj, -1)
    Sx = jnp.concatenate(
        [jnp.broadcast_to(sx, (*batch_shape, sx.shape[-1])) for sx in Sx], -1
    )

    n_col = idx
    n_rhs = len(port_map)

    Cmap = {
        int(instance_ports[k]): int(instance_ports[v]) for k, v in connections.items()
    }
    Ci = jnp.array(list(Cmap.keys()), dtype=jnp.int32)
    Cj = jnp.array(list(Cmap.values()), dtype=jnp.int32)

    Cextmap = {int(instance_ports[k]): int(port_map[v]) for k, v in inverse_ports.items()}
    Cexti = jnp.stack(list(Cextmap.keys()), 0)
    Cextj = jnp.stack(list(Cextmap.values()), 0)
    Cext = jnp.zeros((n_col, n_rhs), dtype=complex).at[Cexti, Cextj].set(1.0)

    # TODO: make this block jittable...
    Ix = jnp.ones((*batch_shape, n_col))
    Ii = Ij = jnp.arange(n_col)
    mask = Cj[None,:] == Si[:, None]
    CSi = jnp.broadcast_to(Ci[None, :], mask.shape)[mask]

    # CSi = jnp.where(Cj[None, :] == Si[:, None], Ci[None, :], 0).sum(1)
    mask = (Cj[:, None] == Si[None, :]).any(0)
    CSj = Sj[mask]
    
    if Sx.ndim > 1: # bug in JAX... see https://github.com/google/jax/issues/9050
        CSx = Sx[..., mask]
    else:
        CSx = Sx[mask]
        
    # CSj = jnp.where(mask, Sj, 0)
    # CSx = jnp.where(mask, Sx, 0.0)

    I_CSi = jnp.concatenate([CSi, Ii], -1)
    I_CSj = jnp.concatenate([CSj, Ij], -1)
    I_CSx = jnp.concatenate([-CSx, Ix], -1)

    n_col, n_rhs = Cext.shape
    n_lhs = jnp.prod(jnp.array(batch_shape, dtype=jnp.int32))
    Sx = Sx.reshape(n_lhs, -1)
    I_CSx = I_CSx.reshape(n_lhs, -1)

    inv_I_CS_Cext = solve_klu(I_CSi, I_CSj, I_CSx, Cext)
    S_inv_I_CS_Cext = mul_coo(Si, Sj, Sx, inv_I_CS_Cext)

    CextT_S_inv_I_CS_Cext = S_inv_I_CS_Cext[..., Cexti, :][..., :, Cextj]
    
    _, n, _ = CextT_S_inv_I_CS_Cext.shape
    S = CextT_S_inv_I_CS_Cext.reshape(*batch_shape, n, n)

    return S, port_map


# ## Example

# hide
if klujax is None:
    def evaluate_circuit_klu(
        instances: Dict[str, SType],
        connections: Dict[str, str],
        ports: Dict[str, str],
    ):
        sdict = evaluate_circuit(instances, connections, ports)
        sdense = sax.sdense(sdict)
        return sdense

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

mzi_sdense: SDense = evaluate_circuit_klu(
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
# -

# the KLU backend yields `SDense` results by default:

mzi_sdense

# An `SDense` is returned for perfomance reasons. By returning an `SDense` by default we prevent any internal `SDict -> SDense` conversions in deeply hierarchical circuits. It's however very easy to convert `SDense` to `SDict` as a final step. To do this, wrap the result (or the function generating the result) with `sdict`:

sax.sdict(mzi_sdense)

# ## Algorithm Walkthrough

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

# Let's first enforce $C^T = C$:

connections = {**connections, **{v: k for k, v in connections.items()}}
connections

# We'll also need the reversed ports:

inverse_ports = {v: k for k, v in ports.items()}
inverse_ports

# An the port indices

port_map = {k: i for i, k in enumerate(ports)}
port_map

# Let's now create the COO-representation of our block diagonal S-matrix $S_{bd}$:

# +
idx, Si, Sj, Sx, instance_ports = 0, [], [], [], {}
batch_shape = ()
for name, instance in instances.items():
    si, sj, sx, ports_map = scoo(instance)
    Si.append(si + idx)
    Sj.append(sj + idx)
    Sx.append(sx)
    if len(sx.shape[:-1]) > len(batch_shape):
        batch_shape = sx.shape[:-1]
    instance_ports.update({f"{name},{p}": i + idx for p, i in ports_map.items()})
    idx += len(ports_map)
Si = jnp.concatenate(Si, -1)
Sj = jnp.concatenate(Sj, -1)
Sx = jnp.concatenate([jnp.broadcast_to(sx, (*batch_shape, sx.shape[-1])) for sx in Sx], -1)

print(Si)
print(Sj)
print(Sx)
# -

# note that we also kept track of the `batch_shape`, i.e. the number of independent simulations (usually number of wavelengths). In the example being used here we don't have a batch dimension (all elements of the `SDict` are `0D`):

batch_shape

# We'll also keep track of the number of columns

n_col = idx
n_col

# And we'll need to solve the circuit for each output port, i.e. we need to solve `n_rhs` number of equations:

n_rhs = len(port_map)
n_rhs

# We can represent the internal connection matrix $C_{int}$ as a mapping between port indices:

Cmap = {int(instance_ports[k]): int(instance_ports[v]) for k, v in connections.items()}
Cmap

# Therefore, the COO-representation of this connection matrix can be obtained as follows (note that an array of values Cx is not necessary, all non-zero elements in a connection matrix are 1)

Ci = jnp.array(list(Cmap.keys()), dtype=jnp.int32)
Cj = jnp.array(list(Cmap.values()), dtype=jnp.int32)
print(Ci)
print(Cj)

# We can represent the external connection matrix $C_{ext}$ as a map between internal port indices and external port indices:

Cextmap = {int(instance_ports[k]): int(port_map[v]) for k, v in inverse_ports.items()}
Cextmap

# Just as for the internal matrix we can represent this external connection matrix in COO-format:

Cexti = jnp.stack(list(Cextmap.keys()), 0)
Cextj = jnp.stack(list(Cextmap.values()), 0)
print(Cexti)
print(Cextj)

# However, we actually need it as a dense representation:
#
# > help needed: can we find a way later on to keep this sparse?

if JAX_AVAILABLE:
    Cext = jnp.zeros((n_col, n_rhs), dtype=complex).at[Cexti, Cextj].set(1.0)
    Cext

# We'll now calculate the row index `CSi` of $C_{int}S_{bd}$ in COO-format:

# TODO: make this block jittable...
Ix = jnp.ones((*batch_shape, n_col))
Ii = Ij = jnp.arange(n_col)
mask = Cj[None,:] == Si[:, None]
CSi = jnp.broadcast_to(Ci[None, :], mask.shape)[mask]
CSi

# > `CSi`: possible jittable alternative? how do we remove the zeros?

CSi_ = jnp.where(Cj[None, :] == Si[:, None], Ci[None, :], 0).sum(1) # not used
CSi_ # not used

# The column index `CSj` of $C_{int}S_{bd}$ can more easily be obtained:

mask = (Cj[:, None] == Si[None, :]).any(0)
CSj = Sj[mask]
CSj

# > `CSj`: possible jittable alternative? how do we remove the zeros?

CSj_ = jnp.where(mask, Sj, 0) # not used
CSj_ # not used

# Finally, the values `CSx` of $C_{int}S_{bd}$ can be obtained as follows:

# +
if Sx.ndim > 1:
    CSx = Sx[..., mask] # normally this should be enough
else:
    CSx = Sx[mask] # need separate case bc bug in JAX... see https://github.com/google/jax/issues/9050
   
CSx
# -

# > `CSx`: possible jittable alternative? how do we remove the zeros?

CSx_ = jnp.where(mask, Sx, 0.0) # not used
CSx_ # not used

# Now we calculate $\mathbb{1} - C_{int}S_{bd}$ in an *uncoalesced* way (we might have duplicate indices on the diagonal):
#
# > **uncoalesced**: having duplicate index combinations (i, j) in the representation possibly with different corresponding values. This is usually not a problem as in linear operations these values will end up to be summed, usually the behavior you want:

I_CSi = jnp.concatenate([CSi, Ii], -1)
I_CSj = jnp.concatenate([CSj, Ij], -1)
I_CSx = jnp.concatenate([-CSx, Ix], -1)
print(I_CSi)
print(I_CSj)
print(I_CSx)

if JAX_AVAILABLE:
    n_col, n_rhs = Cext.shape
    print(n_col, n_rhs)

# The batch shape dimension can generally speaking be anything (in the example here 0D). We need to do the necessary reshapings to make the batch shape 1D:

n_lhs = jnp.prod(jnp.array(batch_shape, dtype=jnp.int32))
print(n_lhs)

Sx = Sx.reshape(n_lhs, -1)
Sx.shape

I_CSx = I_CSx.reshape(n_lhs, -1)
I_CSx.shape

# We're finally ready to do the most important part of the calculation, which we conveniently leave to `klujax` and `SuiteSparse`:

if klujax is not None:
    inv_I_CS_Cext = solve_klu(I_CSi, I_CSj, I_CSx, Cext)
else:
    inv_I_CS_Cext = jnp.array([[[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], [ -0.0 + 0.0j, -0.0 + 0.0j, 0.35355339 + 0.61237569j, -0.61237569 + 0.35355339j, ], [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.70710678j, 0.70710678 + 0.0j], [0.70710678 - 0.0j, -0.0 + 0.70710678j, -0.0 - 0.0j, -0.0 - 0.0j], [-0.0 - 0.0j, -0.0 - 0.0j, 0.70710678 - 0.0j, -0.0 + 0.70710678j], [ 0.35355339 + 0.61237569j, -0.61237569 + 0.35355339j, -0.0 + 0.0j, -0.0 + 0.0j, ], [0.0 + 0.70710678j, 0.70710678 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j]]])

# one more sparse multiplication:

if klujax is not None:
    S_inv_I_CS_Cext = mul_coo(Si, Sj, Sx, inv_I_CS_Cext)
else:
    S_inv_I_CS_Cext = jnp.array([[[0.0 + 0.0j, 0.0 + 0.0j, -0.25 + 0.433015j, -0.433015 + 0.75j], [0.0 + 0.0j, 0.0 + 0.0j, -0.433015 + 0.75j, 0.25 - 0.433015j], [0.70710678 + 0.0j, 0.0 + 0.70710678j, 0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.70710678j, 0.70710678 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], [ 0.0 + 0.0j, 0.0 + 0.0j, 0.35355339 + 0.61237569j, -0.61237569 + 0.35355339j, ], [ 0.35355339 + 0.61237569j, -0.61237569 + 0.35355339j, 0.0 + 0.0j, 0.0 + 0.0j, ], [0.0 + 0.0j, 0.0 + 0.0j, 0.70710678 + 0.0j, 0.0 + 0.70710678j], [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.70710678j, 0.70710678 + 0.0j], [-0.25 + 0.433015j, -0.433015 + 0.75j, 0.0 + 0.0j, 0.0 + 0.0j], [-0.433015 + 0.75j, 0.25 - 0.433015j, 0.0 + 0.0j, 0.0 + 0.0j]]])

# And one more $C_{ext}$ multiplication which we do by clever indexing:

if klujax is not None:
    CextT_S_inv_I_CS_Cext = S_inv_I_CS_Cext[..., Cexti, :][..., :, Cextj]
else:
    CextT_S_inv_I_CS_Cext = jnp.array([[[0.0 + 0.0j, 0.0 + 0.0j, -0.25 + 0.433015j, -0.433015 + 0.75j], [0.0 + 0.0j, 0.0 + 0.0j, -0.433015 + 0.75j, 0.25 - 0.433015j], [-0.25 + 0.433015j, -0.433015 + 0.75j, 0.0 + 0.0j, 0.0 + 0.0j], [-0.433015 + 0.75j, 0.25 - 0.433015j, 0.0 + 0.0j, 0.0 + 0.0j]]])
CextT_S_inv_I_CS_Cext

# That's it! We found the S-matrix of the circuit. We just need to reshape the batch dimension back into the matrix:

_, n, _ = CextT_S_inv_I_CS_Cext.shape
S = CextT_S_inv_I_CS_Cext.reshape(*batch_shape, n, n)
S

# Oh and to complete the `SDense` representation we need to specify the port map as well:

port_map

# ## Algorithm Improvements

# This algorithm is 
#
# * very fast for large circuits 🙂
#
# This algorithm is however:
#
# * **not** jittable 😥
# * **not** differentiable 😥
# * **not** GPU-compatible 🙂
#
# There are probably still plenty of improvements possible for this algorithm:
#
# * **¿** make it jittable **?**
# * **¿** make it differentiable (requires making klujax differentiable first) **?**
# * **¿** make it GPU compatible (requires making suitesparse GPU compatible... probably not gonna happen)**?**
#
# Bottom line is... Do you know how to improve this algorithm or how to implement the above suggestions? Please open a Merge Request!

# ## Debug

instances = {
    "lft": (
        jnp.array(
            [
                [
                    5.19688622e-06 - 1.19777138e-05j,
                    6.30595625e-16 - 1.48061189e-17j,
                    -3.38542541e-01 - 6.15711852e-01j,
                    5.80662654e-03 - 1.11068866e-02j,
                    -3.38542542e-01 - 6.15711852e-01j,
                    -5.80662660e-03 + 1.11068866e-02j,
                ],
                [
                    8.59445189e-16 - 8.29783014e-16j,
                    -2.08640825e-06 + 8.17315497e-06j,
                    2.03847666e-03 - 2.10649131e-03j,
                    5.30509661e-01 + 4.62504708e-01j,
                    -2.03847666e-03 + 2.10649129e-03j,
                    5.30509662e-01 + 4.62504708e-01j,
                ],
                [
                    -3.38542541e-01 - 6.15711852e-01j,
                    2.03847660e-03 - 2.10649129e-03j,
                    7.60088070e-06 + 9.07340423e-07j,
                    2.79292426e-09 + 2.79093547e-07j,
                    5.07842364e-06 + 2.16385350e-06j,
                    -6.84244232e-08 - 5.00486817e-07j,
                ],
                [
                    5.80662707e-03 - 1.11068869e-02j,
                    5.30509661e-01 + 4.62504708e-01j,
                    2.79291895e-09 + 2.79093540e-07j,
                    -4.55645798e-06 + 1.50570403e-06j,
                    6.84244128e-08 + 5.00486817e-07j,
                    -3.55812153e-06 + 4.59781091e-07j,
                ],
                [
                    -3.38542541e-01 - 6.15711852e-01j,
                    -2.03847672e-03 + 2.10649131e-03j,
                    5.07842364e-06 + 2.16385349e-06j,
                    6.84244230e-08 + 5.00486816e-07j,
                    7.60088070e-06 + 9.07340425e-07j,
                    -2.79292467e-09 - 2.79093547e-07j,
                ],
                [
                    -5.80662607e-03 + 1.11068863e-02j,
                    5.30509662e-01 + 4.62504708e-01j,
                    -6.84244296e-08 - 5.00486825e-07j,
                    -3.55812153e-06 + 4.59781093e-07j,
                    -2.79293217e-09 - 2.79093547e-07j,
                    -4.55645798e-06 + 1.50570403e-06j,
                ],
            ]
        ),
        {"in0": 0, "out0": 2, "out1": 4},
    ),
    "top": {("in0", "out0"): -0.99477 - 0.10211j, ("out0", "in0"): -0.99477 - 0.10211j},
    "rgt": {
        ("in0", "out0"): 0.7071067811865476,
        ("in0", "out1"): 0.7071067811865476j,
        ("in1", "out0"): 0.7071067811865476j,
        ("in1", "out1"): 0.7071067811865476,
        ("out0", "in0"): 0.7071067811865476,
        ("out1", "in0"): 0.7071067811865476j,
        ("out0", "in1"): 0.7071067811865476j,
        ("out1", "in1"): 0.7071067811865476,
    },
}
connections = {"lft,out0": "rgt,in0", "lft,out1": "top,in0", "top,out0": "rgt,in1"}
ports = {"in0": "lft,in0", "out0": "rgt,out0"}

sax.sdict(evaluate_circuit(instances, connections, ports))

sax.sdict(evaluate_circuit_klu(instances, connections, ports))
