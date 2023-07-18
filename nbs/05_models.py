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
# default_exp models
# -

# # Models
#
# > Default SAX Models

# hide
import os, sys; sys.stderr = open(os.devnull, "w")

# +
# export
from __future__ import annotations

import warnings
from functools import lru_cache as cache
from types import SimpleNamespace
from typing import Optional, Tuple

import sax
from sax.typing_ import Model, SCoo, SDict
from sax.utils import get_inputs_outputs, reciprocal

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


# -

# ## Standard Models

# +
# export

def straight(
    *,
    wl: float = 1.55,
    wl0: float = 1.55,
    neff: float = 2.34,
    ng: float = 3.4,
    length: float = 10.0,
    loss: float = 0.0
) -> SDict:
    """a simple straight waveguide model"""
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * neff * length / wl
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission =  amplitude * jnp.exp(1j * phase)
    sdict = reciprocal(
        {
            ("in0", "out0"): transmission,
        }
    )
    return sdict


# -

straight()


# +
# export

def coupler(*, coupling: float = 0.5) -> SDict:
    """a simple coupler model"""
    kappa = coupling ** 0.5
    tau = (1 - coupling) ** 0.5
    sdict = reciprocal(
        {
            ("in0", "out0"): tau,
            ("in0", "out1"): 1j * kappa,
            ("in1", "out0"): 1j * kappa,
            ("in1", "out1"): tau,
        }
    )
    return sdict


# -

coupler()


# ## Model Factories

# +
# exporti

def _validate_ports(ports, num_inputs, num_outputs, diagonal) -> Tuple[Tuple[str,...], Tuple[str,...], int, int]:
    if ports is None:
        if num_inputs is None or num_outputs is None:
            raise ValueError(
                "if not ports given, you must specify how many input ports "
                "and how many output ports a model has."
            )
        input_ports = [f"in{i}" for i in range(num_inputs)]
        output_ports = [f"out{i}" for i in range(num_outputs)]
    else:
        if num_inputs is not None:
            if num_outputs is None:
                raise ValueError(
                    "if num_inputs is given, num_outputs should be given as well."
                )
        if num_outputs is not None:
            if num_inputs is None:
                raise ValueError(
                    "if num_outputs is given, num_inputs should be given as well."
                )
        if num_inputs is not None and num_outputs is not None:
            if num_inputs + num_outputs != len(ports):
                raise ValueError("num_inputs + num_outputs != len(ports)")
            input_ports = ports[:num_inputs]
            output_ports = ports[num_inputs:]
        else:
            input_ports, output_ports = get_inputs_outputs(ports)
            num_inputs = len(input_ports)
            num_outputs = len(output_ports)
            
    if diagonal:
        if num_inputs != num_outputs:
            raise ValueError(
                "Can only have a diagonal passthru if number of input ports equals the number of output ports!"
            )
    return input_ports, output_ports, num_inputs, num_outputs


# +
# export

@cache
def unitary(
    num_inputs: Optional[int] = None,
    num_outputs: Optional[int] = None,
    ports: Optional[Tuple[str, ...]] = None,
    *,
    jit=True,
    reciprocal=True,
    diagonal=False,
) -> Model:
    input_ports, output_ports, num_inputs, num_outputs = _validate_ports(ports, num_inputs, num_outputs, diagonal)
    assert num_inputs is not None and num_outputs is not None
    
    # let's create the squared S-matrix:
    N = max(num_inputs, num_outputs)
    S = jnp.zeros((2*N, 2*N), dtype=float)

    if not diagonal:
        if JAX_AVAILABLE:
            S = S.at[:N, N:].set(1)
        else:
            S[:N, N:] = 1
    else:
        r = jnp.arange(N, dtype=int) # reciprocal only works if num_inputs == num_outputs!
        if JAX_AVAILABLE:
            S = S.at[r, N+r].set(1)
        else:
            S[r, N+r] = 1

    if reciprocal:
        if not diagonal:
            if JAX_AVAILABLE:
                S = S.at[N:, :N].set(1)
            else:
                S[N:, :N] = 1
        else:
            r = jnp.arange(N, dtype=int) # reciprocal only works if num_inputs == num_outputs!
            if JAX_AVAILABLE:
                S = S.at[N+r, r].set(1)
            else:
                S[N+r, r] = 1

    # Now we need to normalize the squared S-matrix
    U, s, V = jnp.linalg.svd(S, full_matrices=False)
    S = jnp.sqrt(U@jnp.diag(jnp.where(s > 1e-12, 1, 0))@V)
    
    # Now create subset of this matrix we're interested in:
    r = jnp.concatenate([jnp.arange(num_inputs, dtype=int), N+jnp.arange(num_outputs, dtype=int)], 0)
    S = S[r, :][:, r]

    # let's convert it in SCOO format:
    Si, Sj = jnp.where(S > 1e-6)
    Sx = S[Si, Sj]
    
    # the last missing piece is a port map:
    pm = {
        **{p: i for i, p in enumerate(input_ports)},
        **{p: i + num_inputs for i, p in enumerate(output_ports)},
    }
    
    def func(wl: float = 1.5) -> SCoo:
        wl_ = jnp.asarray(wl)
        Sx_ = jnp.broadcast_to(Sx, (*wl_.shape, *Sx.shape))
        return Si, Sj, Sx_, pm

    func.__name__ = f"unitary_{num_inputs}_{num_outputs}"
    func.__qualname__ = f"unitary_{num_inputs}_{num_outputs}"
    if jit:
        return jax.jit(func)
    return func


# -

# A unitary model returns an `SCoo` by default:

unitary_model = unitary(2, 2)
unitary_model() # a unitary model returns an SCoo by default

# As you probably already know, it's very easy to convert a model returning any `Stype` into a model returning an `SDict` as follows:

unitary_sdict_model = sax.sdict(unitary_model)
unitary_sdict_model()

# If we need custom port names, we can also just specify them explicitly:

unitary_model = unitary(ports=("in0", "in1", "out0", "out1"))
unitary_model()

# A unitary model will by default split a signal at an input port equally over all output ports. However, if there are an equal number of input ports as output ports we can in stead create a passthru by setting the `diagonal` flag to `True`:

passthru_model = unitary(2, 2, diagonal=True)
sax.sdict(passthru_model())

ports_in=['in0']
ports_out=['out0', 'out1', 'out2', 'out3', 'out4']
model = unitary(
    ports=tuple(ports_in+ports_out), jit=True, reciprocal=True
)
model = sax.sdict(model)
model()


# Because this is a pretty common usecase we have a dedicated model factory for this as well. This passthru component just takes the number of links (`'in{i}' -> 'out{i]'`) as input. Alternatively, as before, one can also specify the port names directly but one needs to ensure that `len(ports) == 2*num_links`.

# export
@cache
def passthru(
    num_links: Optional[int] = None,
    ports: Optional[Tuple[str, ...]] = None,
    *,
    jit=True,
    reciprocal=True,
) -> Model:
    passthru = unitary(num_links, num_links, ports, jit=jit, reciprocal=reciprocal, diagonal=True)
    passthru.__name__ = f"passthru_{num_links}_{num_links}"
    passthru.__qualname__ = f"passthru_{num_links}_{num_links}"
    if jit:
        return jax.jit(passthru)
    return passthru


passthru_model = passthru(3)
passthru_sdict_model = sax.sdict(passthru_model)
passthru_sdict_model()

mzi, _ = sax.circuit(
    netlist={
        "instances": {
            "lft": 'u12',
            "top": 'u11',
            "rgt": 'u12',
        },
        "connections": {
            "lft,out0": "rgt,out0",
            "lft,out1": "top,in0",
            "top,out0": "rgt,out1",
        },
        "ports": {
            "in0": "lft,in0",
            "out0": "rgt,in0",
        },
    },
    models={
        'u12': unitary(1, 2),
        'u11': unitary(1, 1),
    },
)
mzi()


# +
# export

@cache
def copier(
    num_inputs: Optional[int] = None,
    num_outputs: Optional[int] = None,
    ports: Optional[Tuple[str, ...]] = None,
    *,
    jit=True,
    reciprocal=True,
    diagonal=False,
) -> Model:
        
    input_ports, output_ports, num_inputs, num_outputs = _validate_ports(ports, num_inputs, num_outputs, diagonal)
    assert num_inputs is not None and num_outputs is not None
    
    # let's create the squared S-matrix:
    S = jnp.zeros((num_inputs+num_outputs, num_inputs+num_outputs), dtype=float)

    if not diagonal:
        if JAX_AVAILABLE:
            S = S.at[:num_inputs, num_inputs:].set(1)
        else:
            S[:num_inputs, num_inputs:] = 1
    else:
        r = jnp.arange(num_inputs, dtype=int) # == range(num_outputs) # reciprocal only works if num_inputs == num_outputs!
        if JAX_AVAILABLE:
            S = S.at[r, num_inputs+r].set(1)
        else:
            S[r, num_inputs+r] = 1

    if reciprocal:
        if not diagonal:
            if JAX_AVAILABLE:
                S = S.at[num_inputs:, :num_inputs].set(1)
            else:
                S[num_inputs:, :num_inputs] = 1
        else:
            r = jnp.arange(num_inputs, dtype=int) # == range(num_outputs) # reciprocal only works if num_inputs == num_outputs!
            if JAX_AVAILABLE:
                S = S.at[num_inputs+r, r].set(1)
            else:
                S[num_inputs+r, r] = 1

    # let's convert it in SCOO format:
    Si, Sj = jnp.where(S > 1e-6)
    Sx = S[Si, Sj]
    
    # the last missing piece is a port map:
    pm = {
        **{p: i for i, p in enumerate(input_ports)},
        **{p: i + num_inputs for i, p in enumerate(output_ports)},
    }
    
    def func(wl: float = 1.5) -> SCoo:
        wl_ = jnp.asarray(wl)
        Sx_ = jnp.broadcast_to(Sx, (*wl_.shape, *Sx.shape))
        return Si, Sj, Sx_, pm

    func.__name__ = f"unitary_{num_inputs}_{num_outputs}"
    func.__qualname__ = f"unitary_{num_inputs}_{num_outputs}"
    if jit:
        return jax.jit(func)
    return func


# -

# A copier model is like a unitary model, but copies the input signal over all output signals. Hence, if the model has multiple output ports, this model can be considered to introduce gain. That said, it can sometimes be a useful component.

copier_model = copier(2, 2)
copier_model() # a copier model returns an SCoo by default

# As you probably already know, it's very easy to convert a model returning any `Stype` into a model returning an `SDict` as follows:

copier_sdict_model = sax.sdict(copier_model)
copier_sdict_model()

# If we need custom port names, we can also just specify them explicitly:

copier_model = copier(ports=("in0", "in1", "out0", "out1"))
copier_model()

ports_in=['in0']
ports_out=['out0', 'out1', 'out2', 'out3', 'out4']
model = unitary(
    ports=tuple(ports_in+ports_out), jit=True, reciprocal=True
)
model = sax.sdict(model)
model()


# Because this is a pretty common usecase we have a dedicated model factory for this as well. This passthru component just takes the number of links (`'in{i}' -> 'out{i]'`) as input. Alternatively, as before, one can also specify the port names directly but one needs to ensure that `len(ports) == 2*num_links`.

# export
@cache
def passthru(
    num_links: Optional[int] = None,
    ports: Optional[Tuple[str, ...]] = None,
    *,
    jit=True,
    reciprocal=True,
) -> Model:
    passthru = unitary(num_links, num_links, ports, jit=jit, reciprocal=reciprocal, diagonal=True)
    passthru.__name__ = f"passthru_{num_links}_{num_links}"
    passthru.__qualname__ = f"passthru_{num_links}_{num_links}"
    if jit:
        return jax.jit(passthru)
    return passthru


passthru_model = passthru(3)
passthru_sdict_model = sax.sdict(passthru_model)
passthru_sdict_model()

# ## All Models

# +
# exports

models = {
    "copier": copier,
    "coupler": coupler,
    "passthru": passthru,
    "straight": straight,
    "unitary": unitary,
}

def get_models(copy: bool=True):
    if copy:
        return {**models}
    return models
