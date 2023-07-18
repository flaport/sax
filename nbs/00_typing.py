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
# default_exp typing_
# -

# # Typing
#
# > SAX types

# +
# hide
import matplotlib.pyplot as plt
from fastcore.test import test_eq
from pytest import approx, raises

import os, sys; sys.stderr = open(os.devnull, "w")

# +
# export
from __future__ import annotations
import functools
import inspect
from collections.abc import Callable as CallableABC
from typing import Any, Callable, Dict, Tuple, Union, cast, overload
try:
    from typing import TypedDict
except ImportError: # python<3.8
    from typing_extensions import TypedDict

import numpy as np
from natsort import natsorted

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False
# -

# ## Core Types

# ### Array

# an `Array` is either a jax array or a numpy array:

# exports
Array = Union[jnp.ndarray, np.ndarray]

# ### Int

# An `Int` is either a built-in `int` or an `Array` [of dtype `int`]

# exports
Int = Union[int, Array]

# ### Float

# A `Float` is eiter a built-in `float` or an `Array` [of dtype `float`]

# exports
Float = Union[float, Array]

# ### ComplexFloat

# A `ComplexFloat` is either a build-in `complex` or an Array [of dtype `complex`]:

# exports
ComplexFloat = Union[complex, Float]

# ### Settings

# A `Settings` dictionary is a nested mapping between setting names [`str`] to either `ComplexFloat` values OR to another lower level `Settings` dictionary.

# exports
Settings = Union[Dict[str, ComplexFloat], Dict[str, "Settings"]]

# Settings dictionaries are used to parametrize a SAX `Model` or a `circuit`. The settings dictionary should have the same hierarchy levels as the circuit:
#  
#  > Example:

mzi_settings = {
    "wl": 1.5,  # global settings
    "lft": {"coupling": 0.5},  # settings for the left coupler
    "top": {"neff": 3.4},  # settings for the top waveguide
    "rgt": {"coupling": 0.3},  # settings for the right coupler
}

# ### SDict

# An `SDict` is a sparse dictionary based representation of an S-matrix, mapping port name tuples such as `('in0', 'out0')` to `ComplexFloat`.

# exports
SDict = Dict[Tuple[str, str], ComplexFloat]

# > Example:

_sdict: SDict = {
    ("in0", "out0"): 3.0,
}
print(_sdict)

# ### SCoo

# An `SCoo` is a sparse matrix based representation of an S-matrix consisting of three arrays and a port map. The three arrays represent the input port indices [`int`], output port indices [`int`] and the S-matrix values [`ComplexFloat`] of the sparse matrix. The port map maps a port name [`str`] to a port index [`int`]. Only these four arrays **together** and in this specific **order** are considered a valid `SCoo` representation!

# exports
SCoo = Tuple[Array, Array, ComplexFloat, Dict[str, int]]

# > Example:

Si = jnp.arange(3, dtype=int)
Sj = jnp.array([0, 1, 0], dtype=int)
Sx = jnp.array([3.0, 4.0, 1.0])
port_map = {"in0": 0, "in1": 2, "out0": 1}
_scoo: SCoo = (Si, Sj, Sx, port_map)
print(Si)
print(Sj)
print(Sx)
print(port_map)

# ### SDense

# an `SDense` is a dense matrix representation of an S-matrix. It's represented by an NxN `ComplexFloat` array and a port map (mapping port names onto port indices).

# exports
SDense = Tuple[Array, Dict[str, int]]

# > Example:

Sd = jnp.arange(9, dtype=float).reshape(3, 3)
port_map = {"in0": 0, "in1": 2, "out0": 1}
_sdense = Sd, port_map
print(Sd)
print(port_map)

# ### SType

# an `SType` is either an `SDict` **OR** an `SCoo` **OR** an `SDense`:

# exports
SType = Union[SDict, SCoo, SDense]

# > Example:

obj: SType = _sdict
obj: SType = _scoo
obj: SType = _sdense

# ### Model

# A `Model` is any keyword-only function that returns an `SType`:

# exports
Model = Callable[..., SType]

# ### ModelFactory

# A `ModelFactory` is any keyword-only function that returns a `Model`:

# exports
ModelFactory = Callable[..., Model]

# > Note: SAX sometimes needs to figure out the difference between a `ModelFactory` and a normal `Model` *before* running the function. To do this, SAX will check the return annotation of the function. Any function with a `-> Model` or `-> Callable` annotation will be considered a `ModelFactory`. Any function without this annotation will be considered a normal Model: **don't forget the return annotation of your Model Factory!** To ensure a correct annotation and to ensure forward compatibility, it's recommended to decorate your `ModelFactory` with the `modelfactory` decorator.

# ### Models

# `Models` is a mapping between model names [`str`] and a `Model`:

# exports
Models = Dict[str, Model]


# > Note: sometimes 'component' is used to refer to a a `Model` or `GeneralModel`. This is because other tools (such as for example GDSFactory) prefer that terminology.

# ## Netlist Types
#
# Netlist types are moved [here](06_netlist.ipynb).

# ## Validation and runtime type-checking:
# > Note: the type-checking functions below are **NOT** very tight and hence should be used within the right context!

# export
def is_float(x: Any) -> bool:
    """Check if an object is a `Float`"""
    if isinstance(x, float):
        return True
    if isinstance(x, np.ndarray):
        return x.dtype in (np.float16, np.float32, np.float64, np.float128)
    if isinstance(x, jnp.ndarray):
        return x.dtype in (jnp.float16, jnp.float32, jnp.float64)
    return False


assert is_float(3.0)
assert not is_float(3)
assert not is_float(3.0 + 2j)
assert not is_float(jnp.array(3.0, dtype=complex))
assert not is_float(jnp.array(3, dtype=int))


# export
def is_complex(x: Any) -> bool:
    """check if an object is a `ComplexFloat`"""
    if isinstance(x, complex):
        return True
    if isinstance(x, np.ndarray):
        return x.dtype in (np.complex64, np.complex128)
    if isinstance(x, jnp.ndarray):
        return x.dtype in (jnp.complex64, jnp.complex128)
    return False


assert not is_complex(3.0)
assert not is_complex(3)
assert is_complex(3.0 + 2j)
assert is_complex(jnp.array(3.0, dtype=complex))
assert not is_complex(jnp.array(3, dtype=int))


# export
def is_complex_float(x: Any) -> bool:
    """check if an object is either a `ComplexFloat` or a `Float`"""
    return is_float(x) or is_complex(x)


assert is_complex_float(3.0)
assert not is_complex_float(3)
assert is_complex_float(3.0 + 2j)
assert is_complex_float(jnp.array(3.0, dtype=complex))
assert not is_complex_float(jnp.array(3, dtype=int))


# export
def is_sdict(x: Any) -> bool:
    """check if an object is an `SDict` (a SAX S-dictionary)"""
    return isinstance(x, dict)


assert not is_sdict(object())
assert is_sdict(_sdict)
assert not is_sdict(_scoo)
assert not is_sdict(_sdense)


# export
def is_scoo(x: Any) -> bool:
    """check if an object is an `SCoo` (a SAX sparse S-matrix representation in COO-format)"""
    return isinstance(x, (tuple, list)) and len(x) == 4


assert not is_scoo(object)
assert not is_scoo(_sdict)
assert is_scoo(_scoo)
assert not is_scoo(_sdense)


# export
def is_sdense(x: Any) -> bool:
    """check if an object is an `SDense` (a SAX dense S-matrix representation)"""
    return isinstance(x, (tuple, list)) and len(x) == 2


assert not is_sdense(object)
assert not is_sdense(_sdict)
assert not is_sdense(_scoo)
assert is_sdense(_sdense)


# +
# export
def is_model(model: Any) -> bool:
    """check if a callable is a `Model` (a callable returning an `SType`)"""
    if not callable(model):
        return False
    try:
        sig = inspect.signature(model)
    except ValueError:
        return False
    for param in sig.parameters.values():
        if param.default is inspect.Parameter.empty:
            return False  # a proper SAX model does not have any positional arguments.
    if _is_callable_annotation(sig.return_annotation):  # model factory
        return False
    return True

def _is_callable_annotation(annotation: Any) -> bool:
    """check if an annotation is `Callable`-like"""
    if isinstance(annotation, str):
        # happens when
        # was imported at the top of the file...
        return annotation.startswith("Callable") or annotation.endswith("Model")
        # TODO: this is not a very robust check...
    try:
        return annotation.__origin__ == CallableABC
    except AttributeError:
        return False


# -

# hide
assert _is_callable_annotation(Callable)
assert not _is_callable_annotation(SDict)


# +
def good_model(x=jnp.array(3.0), y=jnp.array(4.0)) -> SDict:
    return {("in0", "out0"): jnp.array(3.0)}
assert is_model(good_model)

def bad_model(positional_argument, x=jnp.array(3.0), y=jnp.array(4.0)) -> SDict:
    return {("in0", "out0"): jnp.array(3.0)}
assert not is_model(bad_model)


# -

# export
def is_model_factory(model: Any) -> bool:
    """check if a callable is a model function."""
    if not callable(model):
        return False
    sig = inspect.signature(model)
    if _is_callable_annotation(sig.return_annotation):  # model factory
        return True
    return False


# > Note: For a `Callable` to be considered a `ModelFactory` in SAX, it **MUST** have a `Callable` or `Model` return annotation. Otherwise SAX will view it as a `Model` and things might break!

# +
def func() -> Model:
    ...
    
assert is_model_factory(func) # yes, we only check the annotation for now...

def func():
    ...
    
assert not is_model_factory(func) # yes, we only check the annotation for now...


# -

# export
def validate_model(model: Callable):
    """Validate the parameters of a model"""
    positional_arguments = []
    for param in inspect.signature(model).parameters.values():
        if param.default is inspect.Parameter.empty:
            positional_arguments.append(param.name)
    if positional_arguments:
        raise ValueError(
            f"model '{model}' takes positional arguments {', '.join(positional_arguments)} "
            "and hence is not a valid SAX Model! A SAX model should ONLY take keyword arguments (or no arguments at all)."
        )


# +
def good_model(x=jnp.array(3.0), y=jnp.array(4.0)) -> SDict:
    return {("in0", "out0"): jnp.array(3.0)}


assert validate_model(good_model) is None


# +
def bad_model(positional_argument, x=jnp.array(3.0), y=jnp.array(4.0)) -> SDict:
    return {("in0", "out0"): jnp.array(3.0)}


with raises(ValueError):
    validate_model(bad_model)


# -

# export
def is_stype(stype: Any) -> bool:
    """check if an object is an SDict, SCoo or SDense"""
    return is_sdict(stype) or is_scoo(stype) or is_sdense(stype)


# +
# export
def is_singlemode(S: Any) -> bool:
    """check if an stype is single mode"""
    if not is_stype(S):
        return False
    ports = _get_ports(S)
    return not any(("@" in p) for p in ports)

def _get_ports(S: SType):
    if is_sdict(S):
        S = cast(SDict, S)
        ports_set = {p1 for p1, _ in S} | {p2 for _, p2 in S}
        return tuple(natsorted(ports_set))
    else:
        *_, ports_map = S
        assert isinstance(ports_map, dict)
        return tuple(natsorted(ports_map.keys()))


# -

# export
def is_multimode(S: Any) -> bool:
    """check if an stype is single mode"""
    if not is_stype(S):
        return False
    
    ports = _get_ports(S)
    return all(("@" in p) for p in ports)


# export
def is_mixedmode(S: Any) -> bool:
    """check if an stype is neither single mode nor multimode (hence invalid)"""
    return not is_singlemode(S) and not is_multimode(S)


# ## SAX return type helpers
#
# > a.k.a SDict, SDense, SCoo helpers

# Convert an `SDict`, `SCoo` or `SDense` into an `SDict` (or convert a model generating any of these types into a model generating an `SDict`):

# +
# exporti

@overload
def sdict(S: Model) -> Model:
    ...


@overload
def sdict(S: SType) -> SDict:
    ...


# +
# export
def sdict(S: Union[Model, SType]) -> Union[Model, SType]:
    """Convert an `SCoo` or `SDense` to `SDict`"""

    if is_model(S):
        model = cast(Model, S)

        @functools.wraps(model)
        def wrapper(**kwargs):
            return sdict(model(**kwargs))

        return wrapper

    elif is_scoo(S):
        x_dict = _scoo_to_sdict(*cast(SCoo, S))
    elif is_sdense(S):
        x_dict = _sdense_to_sdict(*cast(SDense, S))
    elif is_sdict(S):
        x_dict = cast(SDict, S)
    else:
        raise ValueError("Could not convert arguments to sdict.")

    return x_dict


def _scoo_to_sdict(Si: Array, Sj: Array, Sx: Array, ports_map: Dict[str, int]) -> SDict:
    sdict = {}
    inverse_ports_map = {int(i): p for p, i in ports_map.items()}
    for i, (si, sj) in enumerate(zip(Si, Sj)):
        sdict[
            inverse_ports_map.get(int(si), ""), inverse_ports_map.get(int(sj), "")
        ] = Sx[..., i]
    sdict = {(p1, p2): v for (p1, p2), v in sdict.items() if p1 and p2}
    return sdict


def _sdense_to_sdict(S: Array, ports_map: Dict[str, int]) -> SDict:
    sdict = {}
    for p1, i in ports_map.items():
        for p2, j in ports_map.items():
            sdict[p1, p2] = S[..., i, j]
    return sdict


# -

assert sdict(_sdict) is _sdict
assert sdict(_scoo) == {
    ("in0", "in0"): 3.0,
    ("in1", "in0"): 1.0,
    ("out0", "out0"): 4.0,
}
assert sdict(_sdense) == {
    ("in0", "in0"): 0.0,
    ("in0", "out0"): 1.0,
    ("in0", "in1"): 2.0,
    ("out0", "in0"): 3.0,
    ("out0", "out0"): 4.0,
    ("out0", "in1"): 5.0,
    ("in1", "in0"): 6.0,
    ("in1", "out0"): 7.0,
    ("in1", "in1"): 8.0,
}


# Convert an `SDict`, `SCoo` or `SDense` into an `SCoo` (or convert a model generating any of these types into a model generating an `SCoo`):

# +
# exporti

@overload
def scoo(S: Callable) -> Callable:
    ...


@overload
def scoo(S: SType) -> SCoo:
    ...


# +
# export

def scoo(S: Union[Callable, SType]) -> Union[Callable, SCoo]:
    """Convert an `SDict` or `SDense` to `SCoo`"""

    if is_model(S):
        model = cast(Model, S)

        @functools.wraps(model)
        def wrapper(**kwargs):
            return scoo(model(**kwargs))

        return wrapper

    elif is_scoo(S):
        S = cast(SCoo, S)
    elif is_sdense(S):
        S = _sdense_to_scoo(*cast(SDense, S))
    elif is_sdict(S):
        S = _sdict_to_scoo(cast(SDict, S))
    else:
        raise ValueError("Could not convert arguments to scoo.")

    return S

def _consolidate_sdense(S, pm):
    idxs = list(pm.values())
    S = S[..., idxs, :][..., :, idxs]
    pm = {p: i for i, p in enumerate(pm)}
    return S, pm

def _sdense_to_scoo(S: Array, ports_map: Dict[str, int]) -> SCoo:
    S, ports_map = _consolidate_sdense(S, ports_map)
    Sj, Si = jnp.meshgrid(jnp.arange(S.shape[-1]), jnp.arange(S.shape[-2]))
    return Si.ravel(), Sj.ravel(), S.reshape(*S.shape[:-2], -1), ports_map


def _sdict_to_scoo(sdict: SDict) -> SCoo:
    all_ports = {}
    for p1, p2 in sdict:
        all_ports[p1] = None
        all_ports[p2] = None
    ports_map = {p: i for i, p in enumerate(all_ports)}
    Sx = jnp.stack(jnp.broadcast_arrays(*sdict.values()), -1)
    Si = jnp.array([ports_map[p] for p, _ in sdict])
    Sj = jnp.array([ports_map[p] for _, p in sdict])
    return Si, Sj, Sx, ports_map


# -

scoo(_sdense)

assert scoo(_scoo) is _scoo
assert scoo(_sdict) == (0, 1, 3.0, {"in0": 0, "out0": 1})
Si, Sj, Sx, port_map = scoo(_sdense)  # type: ignore
np.testing.assert_array_equal(Si, jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2]))
np.testing.assert_array_equal(Sj, jnp.array([0, 1, 2, 0, 1, 2, 0, 1, 2]))
np.testing.assert_array_almost_equal(Sx, jnp.array([0.0, 2.0, 1.0, 6.0, 8.0, 7.0, 3.0, 5.0, 4.0]))
assert port_map == {"in0": 0, "in1": 1, "out0": 2}


# Convert an `SDict`, `SCoo` or `SDense` into an `SDense` (or convert a model generating any of these types into a model generating an `SDense`):

# +
# exporti

@overload
def sdense(S: Callable) -> Callable:
    ...


@overload
def sdense(S: SType) -> SDense:
    ...


# +
# export

def sdense(S: Union[Callable, SType]) -> Union[Callable, SDense]:
    """Convert an `SDict` or `SCoo` to `SDense`"""

    if is_model(S):
        model = cast(Model, S)

        @functools.wraps(model)
        def wrapper(**kwargs):
            return sdense(model(**kwargs))

        return wrapper

    if is_sdict(S):
        S = _sdict_to_sdense(cast(SDict, S))
    elif is_scoo(S):
        S = _scoo_to_sdense(*cast(SCoo, S))
    elif is_sdense(S):
        S = cast(SDense, S)
    else:
        raise ValueError("Could not convert arguments to sdense.")

    return S


def _scoo_to_sdense(
    Si: Array, Sj: Array, Sx: Array, ports_map: Dict[str, int]
) -> SDense:
    n_col = len(ports_map)
    S = jnp.zeros((*Sx.shape[:-1], n_col, n_col), dtype=complex)
    if JAX_AVAILABLE:
        S = S.at[..., Si, Sj].add(Sx)
    else:
        S[..., Si, Sj] = Sx
    return S, ports_map

def _sdict_to_sdense(sdict: SDict) -> SDense:
    Si, Sj, Sx, ports_map = _sdict_to_scoo(sdict)
    return _scoo_to_sdense(Si, Sj, Sx, ports_map)


# +
assert sdense(_sdense) is _sdense
Sd, port_map = sdense(_scoo)  # type: ignore
Sd_ = jnp.array([[3.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                 [0.0 + 0.0j, 4.0 + 0.0j, 0.0 + 0.0j],
                 [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]])

np.testing.assert_array_almost_equal(Sd, Sd_)
assert port_map == {"in0": 0, "in1": 2, "out0": 1}


# +
# export

def modelfactory(func):
    """Decorator that marks a function as `ModelFactory`"""
    sig = inspect.signature(func)
    if _is_callable_annotation(sig.return_annotation):  # already model factory
        return func
    func.__signature__ = sig.replace(return_annotation=Model)
    return func
