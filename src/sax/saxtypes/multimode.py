"""SAX Types and type coercions.

Numpy type reference: https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

from __future__ import annotations

__all__ = [
    "ModelFactoryMM",
    "ModelMM",
    "ModelsMM",
    "PortCombinationMM",
    "PortMapMM",
    "PortMode",
    "SCooMM",
    "SCooModelFactoryMM",
    "SCooModelMM",
    "SDenseMM",
    "SDenseModelFactoryMM",
    "SDenseModelMM",
    "SDictMM",
    "SDictModelFactoryMM",
    "SDictModelMM",
    "STypeMM",
]

from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    TypeAlias,
)

from .core import ComplexArray, IntArray1D, Name, val
from .singlemode import (
    cast_string,
    val_callable_annotated,
    val_not_callable_annotated,
    val_port,
    val_sax_callable,
)


def val_mode(obj: Any) -> Mode:
    return str(obj)  # just a string to allow '@0' etc.


Mode: TypeAlias = Annotated[str, val(val_mode)]
"""A mode definition '{mode}'."""


def val_port_mode(obj: Any) -> PortMode:
    s = cast_string(obj)
    parts = s.split("@")
    if len(parts) != 2:
        msg = f"a PortMode should have exactly one '@'-separator. Got: {obj!r}"
        raise TypeError(msg)
    port, mode = parts
    port = val_port(port)
    mode = val_mode(mode)
    return f"{port}@{mode}"


PortMode: TypeAlias = Annotated[str, val(val_port_mode)]
"""A port-mode definition '{port}@{mode}'."""


PortMapMM: TypeAlias = dict[PortMode, int]
"""A mapping from a port to an index."""


PortCombinationMM: TypeAlias = tuple[PortMode, PortMode]
"""A combination of two port names."""


SDictMM: TypeAlias = dict[PortCombinationMM, ComplexArray]
"""A sparse dictionary-based S-matrix representation.

A mapping from a port combination to an S-parameter or an array of S-parameters.

Example:

.. code-block::

    sdict: sax.SDict = {
        ("in0", "out0"): 3.0,
    }

"""

SDenseMM: TypeAlias = tuple[ComplexArray, PortMapMM]
"""A dense S-matrix representation.

S-matrix (2D array) or multidimensional batched S-matrix (N+2)-D array with a port map.
If (N+2)-D array then the S-matrix dimensions are the last two.

Example:

.. code-block::

    Sd = jnp.arange(9, dtype=float).reshape(3, 3)
    port_map = {"in0": 0, "in1": 2, "out0": 1}
    sdense = Sd, port_map

"""

SCooMM: TypeAlias = tuple[IntArray1D, IntArray1D, ComplexArray, PortMapMM]
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

STypeMM: TypeAlias = SDictMM | SCooMM | SDenseMM
"""Any S-Matrix type [SDict, SDense, SCOO]."""


def val_model(model: Any) -> ModelMM:
    return val_not_callable_annotated(val_sax_callable(model))


SDictModelMM: TypeAlias = Annotated[Callable[..., SDictMM], val(val_model)]
"""A keyword-only function producing an SDict."""

SDenseModelMM: TypeAlias = Annotated[Callable[..., SDenseMM], val(val_model)]
"""A keyword-only function producing an SDense."""


SCooModelMM: TypeAlias = Annotated[Callable[..., SCooMM], val(val_model)]
"""A keyword-only function producing an Scoo."""


ModelMM: TypeAlias = Annotated[
    SDictModelMM | SDenseModelMM | SCooModelMM, val(val_model)
]
"""A keyword-only function producing an SType."""


def val_model_factory(model: Any) -> ModelFactoryMM:
    return val_callable_annotated(val_sax_callable(model))


SDictModelFactoryMM: TypeAlias = Annotated[
    Callable[..., SDictModelMM], val(val_model_factory)
]
"""A keyword-only function producing an SDictModel."""


SDenseModelFactoryMM: TypeAlias = Annotated[
    Callable[..., SDenseModelMM], val(val_model_factory)
]
"""A keyword-only function producing an SDenseModel."""

SCooModelFactoryMM: TypeAlias = Annotated[
    Callable[..., SCooModelMM], val(val_model_factory)
]
"""A keyword-only function producing an ScooModel."""


ModelFactoryMM: TypeAlias = Annotated[
    SDictModelFactoryMM | SDenseModelFactoryMM | SCooModelFactoryMM,
    val(val_model_factory),
]
"""A keyword-only function producing a Model."""

ModelsMM: TypeAlias = dict[Name, ModelMM]
"""A mapping between model names and multimode model functions."""
