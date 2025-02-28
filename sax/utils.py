"""General SAX Utilities."""

import re
from collections.abc import Callable
from functools import cache, wraps
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import validate_call

import sax

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


def read(content_or_filename: str | Path | sax.IOLike) -> str:
    """Read the contents of a file."""
    if isinstance(content_or_filename, str) and "\n" in content_or_filename:
        return content_or_filename

    if isinstance(content_or_filename, str | Path):
        return Path(content_or_filename).read_text()

    return content_or_filename.read()


@validate_call(validate_return=True)
def load_netlist(content_or_filename: str | Path | sax.IOLike) -> sax.Netlist:
    """Load a SAX netlist."""
    return yaml.safe_load(read(content_or_filename))


@validate_call(validate_return=True)
def load_recursive_netlist(
    top_level_path: str | Path,
    ext: str = ".pic.yml",
) -> sax.RecursiveNetlist:
    """Load a SAX Recursive Netlist."""
    top_level_path = Path(top_level_path)
    folder_path = top_level_path.parent

    def _net_name(path: Path) -> sax.Name:
        return clean_string(path.name.removesuffix(ext))

    recnet = {_net_name(top_level_path): load_netlist(top_level_path)}

    for path in folder_path.rglob(ext):
        recnet[_net_name(path)] = load_netlist(path)

    return recnet


def clean_string(s: str, dot="p", minus="m", other="_") -> str:
    """Clean a string such that it is a valid python identifier."""
    s = s.strip()
    s = s.replace(".", dot)  # dot
    s = s.replace("-", minus)  # minus
    s = re.sub("[^0-9a-zA-Z_]", other, s)
    if s[0] in "0123456789":
        s = "_" + s
    if not s.isidentifier():
        msg = f"failed to clean string to a valid python identifier: {s}"
        raise ValueError(msg)
    return s
