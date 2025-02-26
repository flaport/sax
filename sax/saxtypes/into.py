"""SAX Types and type coercions.

Numpy type reference: https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

from __future__ import annotations

from collections.abc import Callable
from typing import (
    Any,
    Literal,
    TypeVar,
    _AnnotatedAlias,  # type: ignore[reportAttributeAccessIssue]
    get_args,
    overload,
)

from pydantic import PlainValidator, TypeAdapter

from sax.saxtypes.core import (
    Bool,
    BoolArray,
    BoolArrayLike,
    BoolLike,
    Complex,
    ComplexArray,
    ComplexArray1D,
    ComplexArray1DLike,
    ComplexArrayLike,
    ComplexLike,
    Float,
    FloatArray,
    FloatArray1D,
    FloatArray1DLike,
    FloatArrayLike,
    FloatLike,
    Int,
    IntArray,
    IntArray1D,
    IntArray1DLike,
    IntArrayLike,
    IntLike,
    val_bool,
    val_bool_array,
    val_complex,
    val_complex_array,
    val_complex_array_1d,
    val_float,
    val_float_array,
    val_int,
    val_int_array,
    val_int_array_1d,
)

__all__ = [
    "Into",
]

T = TypeVar("T")


class _IntoMeta(type):
    @overload
    def __getitem__(cls, key: Literal["Bool"]) -> Callable[..., Bool]: ...

    @overload
    def __getitem__(cls, key: Literal["BoolArray"]) -> Callable[..., BoolArray]: ...

    @overload
    def __getitem__(
        cls, key: Literal["BoolArrayLike"]
    ) -> Callable[..., BoolArrayLike]: ...

    @overload
    def __getitem__(cls, key: Literal["BoolLike"]) -> Callable[..., BoolLike]: ...

    @overload
    def __getitem__(cls, key: Literal["Complex"]) -> Callable[..., Complex]: ...

    @overload
    def __getitem__(
        cls, key: Literal["ComplexArray"]
    ) -> Callable[..., ComplexArray]: ...

    @overload
    def __getitem__(
        cls, key: Literal["ComplexArray1D"]
    ) -> Callable[..., ComplexArray1D]: ...

    @overload
    def __getitem__(
        cls, key: Literal["ComplexArray1DLike"]
    ) -> Callable[..., ComplexArray1DLike]: ...

    @overload
    def __getitem__(
        cls, key: Literal["ComplexArrayLike"]
    ) -> Callable[..., ComplexArrayLike]: ...

    @overload
    def __getitem__(cls, key: Literal["ComplexLike"]) -> Callable[..., ComplexLike]: ...

    @overload
    def __getitem__(cls, key: Literal["Float"]) -> Callable[..., Float]: ...

    @overload
    def __getitem__(cls, key: Literal["FloatArray"]) -> Callable[..., FloatArray]: ...

    @overload
    def __getitem__(
        cls, key: Literal["FloatArray1D"]
    ) -> Callable[..., FloatArray1D]: ...

    @overload
    def __getitem__(
        cls, key: Literal["FloatArray1DLike"]
    ) -> Callable[..., FloatArray1DLike]: ...

    @overload
    def __getitem__(
        cls, key: Literal["FloatArrayLike"]
    ) -> Callable[..., FloatArrayLike]: ...

    @overload
    def __getitem__(cls, key: Literal["FloatLike"]) -> Callable[..., FloatLike]: ...

    @overload
    def __getitem__(cls, key: Literal["Int"]) -> Callable[..., Int]: ...

    @overload
    def __getitem__(cls, key: Literal["IntArray"]) -> Callable[..., IntArray]: ...

    @overload
    def __getitem__(cls, key: Literal["IntArray1D"]) -> Callable[..., IntArray1D]: ...

    @overload
    def __getitem__(
        cls, key: Literal["IntArray1DLike"]
    ) -> Callable[..., IntArray1DLike]: ...

    @overload
    def __getitem__(
        cls, key: Literal["IntArrayLike"]
    ) -> Callable[..., IntArrayLike]: ...

    @overload
    def __getitem__(cls, key: Literal["IntLike"]) -> Callable[..., IntLike]: ...

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

    Bool = val_bool
    BoolArray = val_bool_array
    BoolArrayLike = val_bool_array
    BoolLike = val_bool
    Complex = val_complex
    ComplexArray = val_complex_array
    ComplexArray1D = val_complex_array_1d
    ComplexArray1DLike = val_complex_array_1d
    ComplexArrayLike = val_complex_array
    ComplexLike = val_complex
    Float = val_float
    FloatArray = val_float_array
    FloatArray1D = val_complex_array_1d
    FloatArray1DLike = val_complex_array_1d
    FloatArrayLike = val_float_array
    FloatLike = val_float
    Int = val_int
    IntArray = val_int_array
    IntArray1D = val_int_array_1d
    IntArray1DLike = val_int_array_1d
    IntArrayLike = val_int_array
    IntLike = val_int


if __name__ == "__main__":
    x = 3
    y: Float = Into.Float(x)
    print(f"{x} [{type(x)}] {y} [{type(y)}]")
    # print(Int)
    # print(Float)
