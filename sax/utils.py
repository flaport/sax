"""General SAX Utilities."""

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


def maybe(
    func: Callable[..., T], /, exc: type[Exception] = Exception
) -> Callable[..., T | None]:
    """Try a function, return None if it fails."""

    @wraps(func)
    def new_func(*args: Any, **kwargs: Any) -> T | None:  # noqa: ANN401
        try:
            return func(*args, **kwargs)
        except exc:
            return None

    return new_func
