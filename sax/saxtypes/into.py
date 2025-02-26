"""SAX Types and type coercions.

Numpy type reference: https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    _AnnotatedAlias,  # type: ignore[reportAttributeAccessIssue]
    get_args,
    overload,
)

import numpy as np
from pydantic import PlainValidator, TypeAdapter

from sax.saxtypes.core import Float, Int, val_float, val_int

__all__ = [
    "Into",
]

T = TypeVar("T")


class _IntoMeta(type):
    @overload
    def __getitem__(cls, key: Literal["Int"]) -> Callable[..., Int]: ...

    @overload
    def __getitem__(cls, key: Literal["Float"]) -> Callable[..., Float]: ...

    @overload
    def __getitem__(cls, key: type[T]) -> Callable[..., T]: ...

    @overload
    def __getitem__(cls, key: _AnnotatedAlias) -> Callable[..., Any]: ...

    def __getitem__(
        cls, key: type[T] | _AnnotatedAlias | str
    ) -> Callable[..., T | Any]:
        if isinstance(key, str) and hasattr(cls, key):
            return getattr(cls, key)

        if isinstance(key, _AnnotatedAlias):
            validator = get_args(key)[-1]
            if isinstance(validator, PlainValidator):
                return getattr(validator.func, "__wrapped__", validator.func)
        return TypeAdapter(key).validate_python


class Into(metaclass=_IntoMeta):
    """Type caster utility."""

    Int = val_int
    Float = val_float


def _is_bool_param(param: inspect.Parameter | None) -> bool:
    if param is None:
        return False
    return str(param.annotation) == "bool"


if __name__ == "__main__":
    x = 3
    y: Int = Into["Int"](x)
    print(f"{x} [{type(x)}] {y} [{type(y)}]")
    # print(Int)
    # print(Float)
