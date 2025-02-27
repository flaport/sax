"""SAX SingleMode Types and type coercions."""

from __future__ import annotations

__all__ = [
    "InstanceName",
    "InstancePort",
    "ModelFactorySM",
    "ModelSM",
    "Port",
    "PortCombinationSM",
    "PortMapSM",
    "SCooModelFactorySM",
    "SCooModelSM",
    "SCooSM",
    "SDenseModelFactorySM",
    "SDenseModelSM",
    "SDenseSM",
    "SDictModelFactorySM",
    "SDictModelSM",
    "SDictSM",
    "STypeSM",
]

import inspect
from ast import TypeVar
from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    TypeAlias,
    get_origin,
)

from .core import ComplexArrayLike, IntArray1D, val

T = TypeVar("T")


def cast_string(obj: Any) -> str:
    if isinstance(obj, bytes):
        obj = obj.decode()
    return str(obj)


def val_identifier(obj: Any, *, type_name: str) -> str:
    s = cast_string(obj)
    if not s.isidentifier():
        msg = (
            f"A {type_name!r} string should be a valid python identifier. Got: {s!r}. "
            "note: python identifiers should only contain letters, numbers or "
            "underscores. The first character should not be a number."
        )
        raise TypeError(msg)
    return s


def val_instance_name(obj: Any) -> Port:
    return val_identifier(obj, type_name="InstanceName")


InstanceName: TypeAlias = Annotated[str, val(val_instance_name)]


def val_port(obj: Any) -> Port:
    return val_identifier(obj, type_name="Port")


Port: TypeAlias = Annotated[str, val(val_port)]
"""A port definition '{port}'."""


def val_instance_port(obj: Any) -> InstancePort:
    s = cast_string(obj)
    parts = s.split(",")
    if len(parts) > 2:
        msg = f"an InstancePort should have exactly one ','-separator. Got: {obj!r}"
        raise TypeError(msg)
    inst, port = parts
    inst = val_instance_name(inst)
    port = val_port(port)
    return f"{inst},{port}"


InstancePort: TypeAlias = Annotated[str, val(val_instance_port)]
"""An instance port definition '{inst},{port}'."""


PortMapSM: TypeAlias = dict[Port, int]
"""A mapping from a port to an index."""


PortCombinationSM: TypeAlias = tuple[Port, Port]
"""A combination of two port names."""


SDictSM: TypeAlias = dict[PortCombinationSM, ComplexArrayLike]
"""A sparse dictionary-based S-matrix representation.

A mapping from a port combination to an S-parameter or an array of S-parameters.

Example:

.. code-block::

    sdict: sax.SDict = {
        ("in0", "out0"): 3.0,
    }

"""

SDenseSM: TypeAlias = tuple[ComplexArrayLike, PortMapSM]
"""A dense S-matrix representation.

S-matrix (2D array) or multidimensional batched S-matrix (N+2)-D array with a port map.
If (N+2)-D array then the S-matrix dimensions are the last two.

Example:

.. code-block::

    Sd = jnp.arange(9, dtype=float).reshape(3, 3)
    port_map = {"in0": 0, "in1": 2, "out0": 1}
    sdense = Sd, port_map

"""

SCooSM: TypeAlias = tuple[IntArray1D, IntArray1D, ComplexArrayLike, PortMapSM]
"""A sparse S-matrix in COO format (recommended for internal library use only).

An `SCoo` is a sparse matrix based representation of an S-matrix consisting of three
arrays and a port map. The three arrays represent the input port indices [`int`],
output port indices [`int`] and the S-matrix values [`ComplexFloat`] of the sparse
matrix. The port map maps a port name [`str`] to a port index [`int`].

Only these four arrays **together** and in this specific **order** are considered a
valid `SCoo` representation!

Example:

.. code-block::

    Si = jnp.arange(3, dtype=int)
    Sj = jnp.array([0, 1, 0], dtype=int)
    Sx = jnp.array([3.0, 4.0, 1.0])
    port_map = {"in0": 0, "in1": 2, "out0": 1}
    scoo: sax.SCoo = (Si, Sj, Sx, port_map)

Note:
    This representation is only recommended for internal library use. Please don't
    write user-facing code using this representation.

"""

STypeSM: TypeAlias = SDictSM | SCooSM | SDenseSM
"""Any S-Matrix type [SDict, SDense, SCOO]."""


def val_sax_callable(model: Any) -> Callable:
    if not callable(model):
        msg = f"NOT_CALLABLE: A SAX model should be callable. Got: {model!r}."
        raise TypeError(msg)

    model_name = getattr(model, "__name__", str(model))
    try:
        sig = inspect.signature(model)
    except Exception as e:
        msg = (
            f"NO_SIGNATURE: Function '{model_name}' cannot be used as a SAX model. "
            "It has no function signature."
        )
        raise TypeError(msg) from e
    for name, param in sig.parameters.items():
        if param.kind == param.POSITIONAL_ONLY:
            msg = (
                f"NO_POSITIONAL_ONLY: A SAX model should not have positional-only"
                f"arguments. Got: '{model_name}' with param '{name!r}'"
            )
            raise TypeError(msg)
        if param.kind == param.VAR_POSITIONAL:
            msg = (
                f"NO_VAR_POSITIONAL: A SAX model should not have var-positional "
                f"arguments. Got: '{model_name}' with var-param '*{name}'."
            )
            raise TypeError(msg)
        if param.kind == param.VAR_KEYWORD:
            msg = (
                f"NO_VAR_KEYWORD: A SAX model should not have var-keyword "
                f"arguments. Got: '{model_name}' with var-keyword '**{name}'."
            )
            raise TypeError(msg)
        if param.default is inspect.Parameter.empty:
            msg = (
                "NO_DEFAULT: A SAX model should not have arguments without defaults. "
                f"Got: '{model_name}' with param '{name}'."
            )
            raise TypeError(msg)
    return model


def has_callable_return_annotation(model: Callable) -> bool:
    annot = inspect.signature(model).return_annotation
    if isinstance(annot, str) and (
        annot.startswith("Callable") or annot.endswith("Model")
    ):
        return True
    return get_origin(annot) is Callable


def val_not_callable_annotated(model: Callable) -> Callable:
    annot = inspect.signature(model).return_annotation
    if has_callable_return_annotation(model):
        model_name = getattr(model, "__name__", str(model))
        msg = (
            "IS_MODEL_FACTORY: A SAX model should return an SDict, "
            f"SDense, SCoo or SType. Got '{model_name}' returning {annot}. "
            "This indicates that this is in fact a ModelFactory."
        )
        raise TypeError(msg)
    return model


def val_callable_annotated(model: Callable) -> Callable:
    annot = inspect.signature(model).return_annotation
    if not has_callable_return_annotation(model):
        model_name = getattr(model, "__name__", str(model))
        msg = (
            "NOT_ANNOTATED: A SAX ModelFactory should be annotated with a Callable "
            "return annotation to make sure it's not mistaken as a Model. "
            f"Got: '{model_name}' returning {annot}."
        )
        raise TypeError(msg)
    return model


def val_model(model: Any) -> ModelSM:
    return val_not_callable_annotated(val_sax_callable(model))


SDictModelSM: TypeAlias = Annotated[Callable[..., SDictSM], val(val_model)]
"""A keyword-only function producing an SDict."""

SDenseModelSM: TypeAlias = Annotated[Callable[..., SDenseSM], val(val_model)]
"""A keyword-only function producing an SDense."""


SCooModelSM: TypeAlias = Annotated[Callable[..., SCooSM], val(val_model)]
"""A keyword-only function producing an Scoo."""


ModelSM: TypeAlias = Annotated[
    SDictModelSM | SDenseModelSM | SCooModelSM, val(val_model)
]
"""A keyword-only function producing an SType."""


def val_model_factory(model: Any) -> ModelFactorySM:
    return val_callable_annotated(val_sax_callable(model))


SDictModelFactorySM: TypeAlias = Annotated[
    Callable[..., SDictModelSM], val(val_model_factory)
]
"""A keyword-only function producing an SDictModel."""


SDenseModelFactorySM: TypeAlias = Annotated[
    Callable[..., SDenseModelSM], val(val_model_factory)
]
"""A keyword-only function producing an SDenseModel."""

SCooModelFactorySM: TypeAlias = Annotated[
    Callable[..., SCooModelSM], val(val_model_factory)
]
"""A keyword-only function producing an ScooModel."""


ModelFactorySM: TypeAlias = Annotated[
    SDictModelFactorySM | SDenseModelFactorySM | SCooModelFactorySM,
    val(val_model_factory),
]
"""A keyword-only function producing a Model."""
