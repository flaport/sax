"""General SAX Utilities."""

import inspect
import re
from collections.abc import Callable, Iterator
from functools import wraps
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


def clean_string(s: str, dot: str = "p", minus: str = "m", other: str = "_") -> str:
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


def get_settings(model: sax.Model | sax.ModelFactory) -> sax.Settings:
    """Get the parameters of a SAX model function."""
    signature = inspect.signature(model)
    settings: sax.Settings = {
        k: (v.default if not isinstance(v, dict) else v)
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    return sax.into[sax.Settings](settings)


def merge_dicts(*dicts: dict) -> dict:
    """Merge (possibly deeply nested) dictionaries."""
    num_dicts = len(dicts)

    if num_dicts < 1:
        return {}

    if num_dicts == 1:
        return dict(_generate_merged_dict(dicts[0], {}))

    if len(dicts) == 2:  # noqa: PLR2004
        return dict(_generate_merged_dict(dicts[0], dicts[1]))

    return merge_dicts(dicts[0], merge_dicts(*dicts[1:]))


def replace_kwargs(func: Callable, **kwargs: sax.SettingsValue) -> None:
    """Change the kwargs signature of a function."""
    sig = inspect.signature(func)
    settings = [
        inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=v)
        for k, v in kwargs.items()
    ]
    func.__signature__ = sig.replace(parameters=settings)  # type: ignore # noqa: PGH003


def update_settings(
    settings: sax.Settings,
    *compnames: str,
    **kwargs: sax.SettingsValue,
) -> sax.Settings:
    """Update a nested settings dictionary."""
    _settings = {}

    if compnames:
        for k, v in settings.items():
            if isinstance(v, dict):
                if k == compnames[0]:
                    _settings[k] = update_settings(v, *compnames[1:], **kwargs)
                else:
                    _settings[k] = sax.try_into[sax.SettingsValue](v)
            else:
                _settings[k] = sax.try_into[sax.SettingsValue](v)
    else:
        for k, v in settings.items():
            if isinstance(v, dict):
                _settings[k] = update_settings(v, **kwargs)
            elif k in kwargs:
                _settings[k] = sax.try_into[sax.SettingsValue](kwargs[k])
            else:
                _settings[k] = sax.try_into[sax.SettingsValue](v)
    return {k: v for k, v in settings.items() if v is not None}


def _generate_merged_dict(dict1: dict, dict2: dict) -> Iterator[tuple[Any, Any]]:
    # inspired by https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    keys = {
        **{k: None for k in dict1},
        **{k: None for k in dict2},
    }  # keep key order, values irrelevant
    for k in keys:
        if k in dict1 and k in dict2:
            v1, v2 = dict1[k], dict2[k]
            if isinstance(v1, dict) and isinstance(v2, dict):
                v = dict(_generate_merged_dict(v1, v2))
            else:
                # If one of the values is not a dict, you can't continue merging it.
                # Value from second dict overrides one in first and we move on.
                v = v2
        elif k in dict1:
            v = dict1[k]
        else:  # k in dict2:
            v = dict2[k]

        if isinstance(v, dict):
            yield (k, {**v})  # shallow copy of dict
        else:
            yield (k, v)
