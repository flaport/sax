"""SAX Types and type coercions.

Numpy type reference: https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from functools import partial, wraps
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    LiteralString,
    TypeAlias,
    TypeVar,
    cast,
    get_args,
    overload,
)

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from pydantic import PlainValidator
from pydantic_core import PydanticCustomError

from ..utils import maybe

__all__ = [
    "ArrayLike",
    "Bool",
    "BoolArray",
    "BoolArrayLike",
    "BoolLike",
    "Complex",
    "ComplexArray",
    "ComplexArray1D",
    "ComplexArray1DLike",
    "ComplexArrayLike",
    "ComplexLike",
    "Float",
    "FloatArray",
    "FloatArray1D",
    "FloatArray1DLike",
    "FloatArrayLike",
    "FloatLike",
    "InstancePort",
    "Int",
    "IntArray",
    "IntArray1D",
    "IntArray1DLike",
    "IntArrayLike",
    "IntLike",
    "Mode",
    "Model",
    "ModelFactory",
    "Port",
    "PortCombination",
    "PortMap",
    "PortMode",
    "SCoo",
    "SCooModel",
    "SCooModelFactory",
    "SDense",
    "SDenseModel",
    "SDenseModelFactory",
    "SDict",
    "SDictModel",
    "SDictModelFactory",
    "SType",
    "Settings",
    "SettingsValue",
]

T = TypeVar("T")

ArrayLike: TypeAlias = Array | np.ndarray
"""Anything that can turn into an array with ndim>=1."""


def _val(fun: Callable, **val_kwargs: Any) -> PlainValidator:
    @wraps(fun)
    def new(*args: Any, **kwargs: Any) -> Any:
        kwargs.update(val_kwargs)
        try:
            return fun(*args, **kwargs)
        except TypeError as e:
            msg = str(e)
            if ":" in msg:
                subtype, msg = msg.split(":", 1)
            else:
                subtype = ""
            raise PydanticCustomError(
                cast(LiteralString, subtype),
                cast(LiteralString, msg),
            ) from e

    return PlainValidator(new)


def _val_item_type(  # noqa: PLR0913
    obj: Any,
    *,
    strict: bool,
    cast: bool,
    type_cast: Callable[..., T],
    type_def: Any,
    type_name: str,
) -> T:
    item = _val_0d(obj, type_name=type_name).item()
    if not isinstance(item, _get_annotated_type(type_def)):
        arr = maybe(np.asarray)(item)
        if arr is None or not np.can_cast(arr, type_cast, casting="same_kind"):  # type: ignore[reportArgumentType]
            msg = f"NOT_{type_name.upper()}: Cannot validate {obj!r} into {type_name}."
            raise TypeError(msg)
        if strict:
            msg = (
                f"NOT_{type_name.upper()}: Strict validation does not allow casting "
                f"{obj!r} [dtype={arr.dtype}] into {type_name}. "
                f"Note: use {type_name}Like type for less strict type checking."
            )
            raise TypeError(msg)
    if cast:
        return type_cast(item)
    return item


@overload
def val_bool(obj: Any, *, strict: bool = ..., cast: Literal[True] = True) -> Bool: ...


@overload
def val_bool(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> BoolLike: ...


def val_bool(obj: Any, *, strict: bool = False, cast: bool = True) -> BoolLike:
    return _val_item_type(
        obj,
        strict=strict,
        cast=cast,
        type_cast=bool,
        type_def=Bool,
        type_name="Bool",
    )


Bool: TypeAlias = Annotated[bool | np.bool_, _val(val_bool, strict=True)]
"""Any boolean."""


@overload
def val_int(obj: Any, *, strict: bool = ..., cast: Literal[True] = True) -> Int: ...


@overload
def val_int(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> IntLike: ...


def val_int(obj: Any, *, strict: bool = False, cast: bool = True) -> IntLike:
    if strict and maybe(partial(val_bool, strict=True, cast=False))(obj) is not None:
        msg = (
            f"NOT_INT: Strict validation does not allow casting {obj} [bool] into Int. "
        )
        raise TypeError(msg)
    return _val_item_type(
        obj,
        strict=strict,
        cast=cast,
        type_cast=int,
        type_def=Int,
        type_name="Int" if strict else "IntLike",
    )


Int: TypeAlias = Annotated[int | np.signedinteger, _val(val_int, strict=True)]
"""Any signed integer."""


@overload
def val_float(obj: Any, *, strict: bool = ..., cast: Literal[True] = True) -> Float: ...


@overload
def val_float(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> FloatLike: ...


def val_float(obj: Any, *, strict: bool = False, cast: bool = True) -> FloatLike:
    return _val_item_type(
        obj,
        strict=strict,
        cast=cast,
        type_cast=float,
        type_def=Float,
        type_name="Float" if strict else "FloatLike",
    )


Float: TypeAlias = Annotated[float | np.floating, _val(val_float, strict=True)]
"""Any float."""


@overload
def val_complex(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> Complex: ...


@overload
def val_complex(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> ComplexLike: ...


def val_complex(obj: Any, *, strict: bool = False, cast: bool = True) -> ComplexLike:
    return _val_item_type(
        obj,
        strict=strict,
        cast=cast,
        type_cast=complex,
        type_def=Complex,
        type_name="Complex" if strict else "ComplexLike",
    )


Complex: TypeAlias = Annotated[
    complex | np.complexfloating, _val(val_complex, strict=True)
]
"""Any complex number."""


def _val_array_type(  # noqa: C901,PLR0913
    obj: Any,
    *,
    strict: bool,
    cast: bool,
    type_def: Any,
    default_dtype: Any,
    type_name: str,
) -> Array:
    if strict:
        if isinstance(obj, np.ndarray):
            msg = (
                f"NOT_{type_name.upper()}: Strict validation does not allow casting "
                f"a numpy array {obj!r} {type(obj)} into a jax-array of "
                f"type {type_name}. Note: use {type_name}Like type for less "
                "strict type checking."
            )
            raise TypeError(msg)
        if not isinstance(obj, Array):
            msg = (
                f"NOT_{type_name.upper()}: Strict validation does not allow casting "
                f"the scalar {obj!r} {type(obj)} into an array type [{type_name}]. "
                f"Note: use {type_name}Like type for less strict type checking."
            )
            raise TypeError(msg)

    short_message = f"Cannot validate {obj!r} into a JAX array."
    arr = _val_array(obj, error_message=short_message)
    ndim = _get_annotated_ndim(type_def)
    if ndim is not None and arr.ndim != ndim:
        if strict:
            msg = (
                f"NOT_{type_name.upper()}: Strict validation does not allow "
                f"broadcasting the {int(arr.ndim)}D array {arr!r} into a {ndim}D "
                f"array. Note: use {type_name}Like type for less strict type checking."
            )
            raise TypeError(msg)
        _ndim = int(arr.ndim)
        while arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim > ndim:
            msg = (
                f"NOT_{type_name.upper()}: Unable to cast the {_ndim}D "
                f"array {obj!r} into a {ndim}D array. "
            )
            raise TypeError(msg)
        while arr.ndim < ndim:
            arr = arr[None]
    if not np.issubdtype(arr.dtype, _get_annotated_dtype(type_def)):
        if not np.can_cast(arr, default_dtype, casting="same_kind"):
            msg = f"NOT_{type_name.upper()}: Cannot validate {obj!r} into {type_name}."
            raise TypeError(msg)
        if strict:
            msg = (
                f"NOT_{type_name.upper()}: Strict validation does not allow casting "
                f"{obj!r} [dtype={arr.dtype}] into {type_name}. "
                f"Note: use {type_name}Like type for less strict type checking."
            )
            raise TypeError(msg)
    if cast:
        return jnp.asarray(arr, dtype=default_dtype)
    return arr


@overload
def val_bool_array(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> BoolArray: ...


@overload
def val_bool_array(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> BoolArrayLike: ...


def val_bool_array(
    obj: Any, *, strict: bool = False, cast: bool = True
) -> BoolArrayLike:
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=BoolArray if strict else BoolArrayLike,
        default_dtype=np.bool_,
        type_name="BoolArray" if strict else "BoolArrayLike",
    )


BoolArray: TypeAlias = Annotated[Array, np.bool_, _val(val_bool_array, strict=True)]
"""N-dimensional Bool array."""


@overload
def val_int_array(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> IntArray: ...


@overload
def val_int_array(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> IntArrayLike: ...


def val_int_array(obj: Any, *, strict: bool = False, cast: bool = True) -> IntArrayLike:
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=IntArray if strict else IntArrayLike,
        default_dtype=np.int64 if _x64_enabled() else np.int32,
        type_name="IntArray" if strict else "IntArrayLike",
    )


IntArray: TypeAlias = Annotated[
    Array, np.signedinteger, _val(val_int_array, strict=True)
]
"""N-dimensional Int array."""


@overload
def val_float_array(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> FloatArray: ...


@overload
def val_float_array(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> FloatArrayLike: ...


def val_float_array(
    obj: Any, *, strict: bool = False, cast: bool = True
) -> FloatArrayLike:
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=FloatArray if strict else FloatArrayLike,
        default_dtype=np.float64 if _x64_enabled() else np.float32,
        type_name="FloatArray" if strict else "FloatArrayLike",
    )


FloatArray: TypeAlias = Annotated[
    Array, np.floating, _val(val_float_array, strict=True)
]
"""N-dimensional Float array."""


@overload
def val_complex_array(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> ComplexArray: ...


@overload
def val_complex_array(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> ComplexArrayLike: ...


def val_complex_array(
    obj: Any, *, strict: bool = False, cast: bool = True
) -> ComplexArrayLike:
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=ComplexArray if strict else ComplexArrayLike,
        default_dtype=np.complex128 if _x64_enabled() else np.complex64,
        type_name="ComplexArray" if strict else "ComplexArrayLike",
    )


ComplexArray: TypeAlias = Annotated[
    Array, np.complexfloating, _val(val_complex_array, strict=True)
]
"""N-dimensional Complex array."""


@overload
def val_int_array_1d(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> IntArray1D: ...


@overload
def val_int_array_1d(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> IntArray1DLike: ...


def val_int_array_1d(
    obj: Any, *, strict: bool = False, cast: bool = True
) -> IntArray1DLike:
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=IntArray1D if strict else IntArray1DLike,
        default_dtype=np.int64 if _x64_enabled() else np.int32,
        type_name="IntArray1D" if strict else "Intarray1DLike",
    )


IntArray1D: TypeAlias = Annotated[
    ArrayLike, np.signedinteger, 1, _val(val_int_array_1d, strict=True)
]
"""1-dimensional Int array."""


@overload
def val_float_array_1d(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> FloatArray1D: ...


@overload
def val_float_array_1d(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> FloatArray1DLike: ...


def val_float_array_1d(
    obj: Any, *, strict: bool = False, cast: bool = True
) -> FloatArray1DLike:
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=FloatArray1D if strict else FloatArray1DLike,
        default_dtype=np.float64 if _x64_enabled() else np.float32,
        type_name="FloatArray1D" if strict else "FloatArray1DLike",
    )


FloatArray1D: TypeAlias = Annotated[
    ArrayLike, np.floating, 1, _val(val_float_array_1d, strict=True)
]
"""1-dimensional Float array."""


@overload
def val_complex_array_1d(
    obj: Any, *, strict: bool = ..., cast: Literal[True] = True
) -> ComplexArray1D: ...


@overload
def val_complex_array_1d(
    obj: Any, *, strict: bool = ..., cast: Literal[False] = False
) -> ComplexArray1DLike: ...


def val_complex_array_1d(
    obj: Any, *, strict: bool = False, cast: bool = True
) -> ComplexArray1DLike:
    return _val_array_type(
        obj,
        strict=strict,
        cast=cast,
        type_def=ComplexArray1D if strict else ComplexArray1DLike,
        default_dtype=np.complex128 if _x64_enabled() else np.complex64,
        type_name="ComplexArray1D" if strict else "ComplexArray1DLike",
    )


ComplexArray1D: TypeAlias = Annotated[
    ArrayLike, np.complexfloating, 1, _val(val_complex_array_1d, strict=True)
]
"""1-dimensional Complex array."""

BoolLike: TypeAlias = Annotated[bool | np.bool_, _val(val_bool, cast=False)]
"""Anything that can be cast into an Int without loss of data."""

IntLike: TypeAlias = Annotated[int | np.integer, _val(val_int, cast=False)]
"""Anything that can be cast into an Int without loss of data."""

FloatLike: TypeAlias = Annotated[
    IntLike | float | np.floating, _val(val_float, cast=False)
]
"""Anything that can be cast into a Float without loss of data."""

ComplexLike: TypeAlias = Annotated[
    FloatLike | complex | np.inexact, _val(val_complex, cast=False)
]
"""Anything that can be cast into a Complex without loss of data."""

BoolArrayLike: TypeAlias = Annotated[
    ArrayLike | BoolLike, np.bool_, _val(val_bool_array, cast=False)
]
"""Anything that can be cast into a N-dimensional Int array without loss of data."""

IntArrayLike: TypeAlias = Annotated[
    ArrayLike | IntLike, np.integer, _val(val_int_array, cast=False)
]
"""Anything that can be cast into a N-dimensional Int array without loss of data."""

FloatArrayLike: TypeAlias = Annotated[
    ArrayLike | FloatLike, np.floating, _val(val_float_array, cast=False)
]
"""Anything that can be cast into a N-dimensional Float array without loss of data."""

ComplexArrayLike: TypeAlias = Annotated[
    ArrayLike | ComplexLike, np.inexact, _val(val_complex_array, cast=False)
]
"""Anything that can be cast into a N-dimensional Complex array without loss of data."""

IntArray1DLike: TypeAlias = Annotated[
    IntArrayLike, np.integer, 1, _val(val_int_array_1d, cast=False)
]
"""1-dimensional integer array."""

FloatArray1DLike: TypeAlias = Annotated[
    FloatArrayLike, np.floating, 1, _val(val_float_array_1d, cast=False)
]
"""1-dimensional float array."""

ComplexArray1DLike: TypeAlias = Annotated[
    ComplexArrayLike, np.inexact, 1, _val(val_complex_array_1d, cast=False)
]
"""1-dimensional complex array."""

Port: TypeAlias = str
"""A port definition '{port}'."""

Mode: TypeAlias = str
"""A mode definition '{mode}'."""

PortMode: TypeAlias = str
"""A port-mode definition '{port}@{mode}'."""

InstancePort: TypeAlias = str
"""An instance port definition '{inst},{port}'."""

PortMap: TypeAlias = dict[Port, int]
"""A mapping from a port to an index."""

PortCombination: TypeAlias = tuple[InstancePort, InstancePort]
"""A combination of two port names (str, str)."""

SDict: TypeAlias = dict[PortCombination, ComplexArrayLike]
"""A sparse dictionary-based S-matrix representation.

A mapping from a port combination to an S-parameter or an array of S-parameters.

Example:

.. code-block::

    sdict: sax.SDict = {
        ("in0", "out0"): 3.0,
    }

"""

SDense: TypeAlias = tuple[ComplexArrayLike, PortMap]
"""A dense S-matrix representation.

S-matrix (2D array) or multidimensional batched S-matrix (N+2)-D array with a port map.
If (N+2)-D array then the S-matrix dimensions are the last two.

Example:

.. code-block::

    Sd = jnp.arange(9, dtype=float).reshape(3, 3)
    port_map = {"in0": 0, "in1": 2, "out0": 1}
    sdense = Sd, port_map

"""

SCoo: TypeAlias = tuple[IntArray1D, IntArray1D, ComplexArrayLike, PortMap]
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

Settings: TypeAlias = dict[str, "SettingsValue"]
"""A (possibly nested) settings mapping.

Example:

.. code-block::

    mzi_settings: sax.Settings = {
        "wl": 1.5,  # global settings
        "lft": {"coupling": 0.5},  # settings for the left coupler
        "top": {"neff": 3.4},  # settings for the top waveguide
        "rgt": {"coupling": 0.3},  # settings for the right coupler
    }

"""

SettingsValue: TypeAlias = Settings | ComplexArrayLike | str | None
"""Anything that can be used as value in a settings dict."""

SType: TypeAlias = SDict | SCoo | SDense
"""Any S-Matrix type [SDict, SDense, SCOO]."""


SDictModel: TypeAlias = Callable[..., SDict]
"""A keyword-only function producing an SDict."""

SDenseModel: TypeAlias = Callable[..., SDense]
"""A keyword-only function producing an SDense."""


SCooModel: TypeAlias = Callable[..., SCoo]
"""A keyword-only function producing an Scoo."""


Model: TypeAlias = SDictModel | SDenseModel | SCooModel
"""A keyword-only function producing an SType."""

SDictModelFactory: TypeAlias = Callable[..., SDictModel]
"""A keyword-only function producing an SDictModel."""


SDenseModelFactory: TypeAlias = Callable[..., SDenseModel]
"""A keyword-only function producing an SDenseModel."""

SCooModelFactory: TypeAlias = Callable[..., SCooModel]
"""A keyword-only function producing an ScooModel."""


ModelFactory: TypeAlias = SDictModelFactory | SDenseModelFactory | SCooModelFactory
"""A keyword-only function producing a Model."""


def _get_annotated_type(annotated: Annotated) -> type | UnionType:
    return get_args(annotated)[0]


def _get_annotated_dtype(annotated: Annotated) -> Any:
    return get_args(annotated)[1]


def _get_annotated_ndim(annotated: Annotated) -> Any:
    with suppress(Exception):
        return int(get_args(annotated)[2])
    return None


def _val_array(obj: Any, error_message: str = "") -> Array:
    try:
        return jnp.asarray(obj)
    except TypeError as e:
        msg = f"NOT_ARRAYLIKE: {error_message}."
        raise TypeError(msg) from e


def _val_0d(obj: Any, *, type_name: str = "0D") -> Array:
    short_message = f"Cannot cast {obj!r} to {type_name}"
    arr = _val_array(obj)
    if arr.ndim > 0:
        msg = (
            f"NOT_SCALAR: {short_message}. The given item should be a scalar. "
            f"Got shape={arr.shape}."
        )
        raise TypeError(msg)
    return arr


def _x64_enabled() -> bool:
    return bool(getattr(jax.config, "jax_enable_x64", False))
