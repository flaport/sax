"""Easily convert a similar type into the desired type.

Example:
    >>> x: int = 5;
    >>> y: sax.Complex = sax.into[sax.Complex](x)
    >>> print(y)
    (5+0j)
    >>> if (z:=sax.try_into[sax.Float](x)) is not None: print(z)
    5.0

"""

from __future__ import annotations

__all__ = ["into", "try_into"]

from collections.abc import Callable
from typing import Any, TypeVar, cast, get_args, overload

from pydantic import PlainValidator, TypeAdapter
from pydantic_core._pydantic_core import ValidationError
from typing_extensions import _AnnotatedAlias

from sax.utils import maybe

T = TypeVar("T")


class Into(type):
    @overload
    def __getitem__(cls, key: type[T]) -> Callable[..., T]: ...

    @overload
    def __getitem__(cls, key: str) -> Callable[..., Any]: ...

    @overload
    def __getitem__(cls, key: Any) -> Callable[..., Any]: ...

    def __getitem__(
        cls,
        key: type[T] | str | Any,
    ) -> Callable[..., T | Any]:
        if isinstance(key, str) and hasattr(cls, key):
            return getattr(cls, key)

        if isinstance(key, _AnnotatedAlias):
            validator = get_args(key)[-1]
            if isinstance(validator, PlainValidator):
                return getattr(validator.func, "__wrapped__", validator.func)
        return _wrapped_validator(cast(type[T], key))


def _wrapped_validator(key: type[T]) -> Callable[..., T]:
    def into_type(obj: Any, /, *, strict: bool | None = None) -> T:
        try:
            return TypeAdapter(key).validate_python(obj, strict=strict)
        except ValidationError as e:
            raise TypeError(str(e)) from e

    return into_type


class into(metaclass=Into):  # noqa: N801
    """Type caster utility."""


class TryInto(type):
    @overload
    def __getitem__(cls, key: type[T]) -> Callable[..., T | None]: ...

    @overload
    def __getitem__(cls, key: str) -> Callable[..., Any | None]: ...

    @overload
    def __getitem__(cls, key: Any) -> Callable[..., Any | None]: ...

    def __getitem__(cls, key: type[T] | str | Any) -> Callable[..., T | Any | None]:
        return maybe(into[key])


class try_into(metaclass=TryInto):  # noqa: N801
    """Type caster utility."""


if __name__ == "__main__":
    from .core import Float, Int

    x = 3
    y: Int = into[Float](x)
