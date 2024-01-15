""" SAX Types and type coercions """

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable as CallableABC
from typing import Any, Callable, Dict, Tuple, Union, cast, overload

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array as Array
from jaxtyping import ArrayLike as ArrayLike
from jaxtyping import Complex as Complex
from jaxtyping import Float as Float
from jaxtyping import Int as Int
from natsort import natsorted

IntArray1D = Int[Array, " dim"]
""" One dimensional integer array """

FloatArray1D = Complex[Array, " dim"]
""" One dimensional float array """

ComplexArray1D = Complex[Array, " dim"]
""" One dimensional complex array """

IntArrayND = Int[Array, "..."]
""" N-dimensional integer array """

FloatArrayND = Complex[Array, "..."]
""" N-dimensional float array """

ComplexArrayND = Complex[Array, "..."]
""" N-dimensional complex array """

PortMap = Dict[str, int]
""" A mapping from a port name (str) to a port index (int) """

PortCombination = Tuple[str, str]
""" A combination of two port names (str, str) """

SDict = Dict[PortCombination, ComplexArrayND]
""" A mapping from a port combination to an S-parameter or an array of S-parameters

Example:

.. code-block::

    sdict: sax.SDict = {
        ("in0", "out0"): 3.0,
    }

"""

SDense = Tuple[ComplexArrayND, PortMap]
""" A dense S-matrix (2D array) or multidimensional batched S-matrix (N+2)-D array
combined with a port map. If (N+2)-D array the S-matrix dimensions are the last two.

Example:

.. code-block::

    Sd = jnp.arange(9, dtype=float).reshape(3, 3)
    port_map = {"in0": 0, "in1": 2, "out0": 1}
    sdense = Sd, port_map

"""

SCoo = Tuple[IntArray1D, IntArray1D, ComplexArrayND, PortMap]
""" A sparse S-matrix in COO format (recommended for internal library use only)

An `SCoo` is a sparse matrix based representation of an S-matrix consisting of three arrays and a port map.
The three arrays represent the input port indices [`int`], output port indices [`int`] and the S-matrix values [`ComplexFloat`] of the sparse matrix.
The port map maps a port name [`str`] to a port index [`int`]. Only these four arrays **together** and in this specific **order** are considered a valid `SCoo` representation!

Example:

.. code-block::

    Si = jnp.arange(3, dtype=int)
    Sj = jnp.array([0, 1, 0], dtype=int)
    Sx = jnp.array([3.0, 4.0, 1.0])
    port_map = {"in0": 0, "in1": 2, "out0": 1}
    scoo: sax.SCoo = (Si, Sj, Sx, port_map)

"""

Settings = Dict[str, Union["Settings", FloatArrayND, ComplexArrayND]]
""" A (possibly recursive) mapping from a setting name to a float or complex value or array

Example:

.. code-block::

    mzi_settings: sax.Settings = {
        "wl": 1.5,  # global settings
        "lft": {"coupling": 0.5},  # settings for the left coupler
        "top": {"neff": 3.4},  # settings for the top waveguide
        "rgt": {"coupling": 0.3},  # settings for the right coupler
    }

"""

SType = Union[SDict, SCoo, SDense]
""" An SDict, SDense or SCOO """

Model = Callable[..., SType]
""" A keyword-only function producing an SType """

ModelFactory = Callable[..., Model]
""" A keyword-only function producing a Model """


def is_float(x: Any) -> bool:
    """Check if an object is a `Float`"""
    if isinstance(x, float):
        return True
    if isinstance(x, np.ndarray):
        return x.dtype in (np.float16, np.float32, np.float64, np.float128)
    if isinstance(x, jnp.ndarray):
        return x.dtype in (jnp.float16, jnp.float32, jnp.float64)
    return False


def is_complex(x: Any) -> bool:
    """check if an object is a `ComplexFloat`"""
    if isinstance(x, complex):
        return True
    if isinstance(x, np.ndarray):
        return x.dtype in (np.complex64, np.complex128)
    if isinstance(x, jnp.ndarray):
        return x.dtype in (jnp.complex64, jnp.complex128)
    return False


def is_complex_float(x: Any) -> bool:
    """check if an object is either a `ComplexFloat` or a `Float`"""
    return is_float(x) or is_complex(x)


def is_sdict(x: Any) -> bool:
    """check if an object is an `SDict` (a SAX S-dictionary)"""
    return isinstance(x, dict)


def is_scoo(x: Any) -> bool:
    """check if an object is an `SCoo` (a SAX sparse S representation in COO-format)"""
    return isinstance(x, (tuple, list)) and len(x) == 4


def is_sdense(x: Any) -> bool:
    """check if an object is an `SDense` (a SAX dense S-matrix representation)"""
    return isinstance(x, (tuple, list)) and len(x) == 2


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


def is_model_factory(model: Any) -> bool:
    """check if a callable is a model function."""
    if not callable(model):
        return False
    sig = inspect.signature(model)
    if _is_callable_annotation(sig.return_annotation):  # model factory
        return True
    return False


def validate_model(model: Callable):
    """Validate the parameters of a model"""
    positional_arguments = []
    for param in inspect.signature(model).parameters.values():
        if param.default is inspect.Parameter.empty:
            positional_arguments.append(param.name)
    if positional_arguments:
        raise ValueError(
            f"model '{model}' takes positional "
            f"arguments {', '.join(positional_arguments)} "
            "and hence is not a valid SAX Model! "
            "A SAX model should ONLY take keyword arguments (or no arguments at all)."
        )


def is_stype(stype: Any) -> bool:
    """check if an object is an SDict, SCoo or SDense"""
    return is_sdict(stype) or is_scoo(stype) or is_sdense(stype)


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


def is_multimode(S: Any) -> bool:
    """check if an stype is single mode"""
    if not is_stype(S):
        return False

    ports = _get_ports(S)
    return all(("@" in p) for p in ports)


def is_mixedmode(S: Any) -> bool:
    """check if an stype is neither single mode nor multimode (hence invalid)"""
    return not is_singlemode(S) and not is_multimode(S)


@overload
def sdict(S: Model) -> Model:
    ...


@overload
def sdict(S: SType) -> SDict:
    ...


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


def _scoo_to_sdict(
    Si: IntArray1D,
    Sj: IntArray1D,
    Sx: ComplexArrayND,
    ports_map: Dict[str, int],
) -> SDict:
    sdict = {}
    inverse_ports_map = {int(i): p for p, i in ports_map.items()}
    for i, (si, sj) in enumerate(zip(Si, Sj)):
        input_port = inverse_ports_map.get(int(si), "")
        output_port = inverse_ports_map.get(int(sj), "")
        sdict[input_port, output_port] = Sx[..., i]
    sdict = {(p1, p2): v for (p1, p2), v in sdict.items() if p1 and p2}
    return sdict


def _sdense_to_sdict(S: Array, ports_map: Dict[str, int]) -> SDict:
    sdict = {}
    for p1, i in ports_map.items():
        for p2, j in ports_map.items():
            sdict[p1, p2] = S[..., i, j]
    return sdict


@overload
def scoo(S: Callable) -> Callable:
    ...


@overload
def scoo(S: SType) -> SCoo:
    ...


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
    ports_map = {p: int(i) for i, p in enumerate(all_ports)}
    Sx = jnp.stack(jnp.broadcast_arrays(*sdict.values()), -1)
    Si = jnp.array([ports_map[p] for p, _ in sdict])
    Sj = jnp.array([ports_map[p] for _, p in sdict])
    return Si, Sj, Sx, ports_map


@overload
def sdense(S: Callable) -> Callable:
    ...


@overload
def sdense(S: SType) -> SDense:
    ...


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
    S = S.at[..., Si, Sj].add(Sx)
    return S, ports_map


def _sdict_to_sdense(sdict: SDict) -> SDense:
    Si, Sj, Sx, ports_map = _sdict_to_scoo(sdict)
    return _scoo_to_sdense(Si, Sj, Sx, ports_map)


def modelfactory(func):
    """Decorator that marks a function as `ModelFactory`"""
    sig = inspect.signature(func)
    if _is_callable_annotation(sig.return_annotation):  # already model factory
        return func
    func.__signature__ = sig.replace(return_annotation=Model)
    return func
