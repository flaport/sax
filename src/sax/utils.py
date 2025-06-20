"""General SAX Utilities."""

from __future__ import annotations

import inspect
import re
import warnings
from collections.abc import Callable, Iterator
from copy import deepcopy
from functools import partial, wraps
from hashlib import md5
from pathlib import Path
from typing import Any, NamedTuple, TypeVar, cast, overload

import jax
import jax.numpy as jnp
import numpy as np
import orjson
import yaml
from numpy.exceptions import ComplexWarning

import sax

T = TypeVar("T")

__all__ = [
    "clean_string",
    "copy_settings",
    "flatten_dict",
    "get_settings",
    "grouped_interp",
    "hash_dict",
    "load_netlist",
    "load_recursive_netlist",
    "maybe",
    "merge_dicts",
    "read",
    "rename_params",
    "rename_ports",
    "replace_kwargs",
    "unflatten_dict",
    "update_settings",
]


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


def copy_settings(settings: sax.Settings) -> sax.Settings:
    """Copy a parameter dictionary."""
    return deepcopy(settings)


def read(content_or_filename: str | Path | sax.IOLike) -> str:
    """Read the contents of a file."""
    if isinstance(content_or_filename, str) and "\n" in content_or_filename:
        return content_or_filename

    if isinstance(content_or_filename, str | Path):
        return Path(content_or_filename).read_text()

    return content_or_filename.read()


def load_netlist(content_or_filename: str | Path | sax.IOLike) -> sax.Netlist:
    """Load a SAX netlist."""
    return yaml.safe_load(read(content_or_filename))


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

    if len(dicts) == 2:
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
    settings: sax.Settings, *compnames: str, **kwargs: sax.SettingsValue
) -> sax.Settings:
    """Update a nested settings dictionary.

    .. note ::

        1. Even though it's possible to update parameter dictionaries in place,
        this function is convenient to apply certain parameters (e.g. wavelength
        'wl' or temperature 'T') globally.
        2. This operation never updates the given settings dictionary inplace.
        3. Any non-float keyword arguments will be silently ignored.

    """
    _settings = {}
    if not compnames:
        for k, v in settings.items():
            if isinstance(v, dict):
                _settings[k] = update_settings(v, **kwargs)
            elif k in kwargs:
                _settings[k] = _try_complex_float(kwargs[k])
            else:
                _settings[k] = _try_complex_float(v)
    else:
        for k, v in settings.items():
            if isinstance(v, dict):
                if k == compnames[0]:
                    _settings[k] = update_settings(v, *compnames[1:], **kwargs)
                else:
                    _settings[k] = v
            else:
                _settings[k] = _try_complex_float(v)
    return _settings


def flatten_dict(dic: dict[str, Any], sep: str = ",") -> dict[str, Any]:
    """Flatten a nested dictionary."""
    return _flatten_dict(dic, sep=sep)


def unflatten_dict(dic: dict[str, Any], sep: str = ",") -> dict[str, Any]:
    """Unflatten a flattened dictionary."""
    # from: https://gist.github.com/fmder/494aaa2dd6f8c428cede
    items = {}

    for k, v in dic.items():
        keys = k.split(sep)
        sub_items = items
        for ki in keys[:-1]:
            if ki in sub_items:
                sub_items = sub_items[ki]
            else:
                sub_items[ki] = {}
                sub_items = sub_items[ki]

        sub_items[keys[-1]] = v

    return items


def grouped_interp(
    wl: sax.FloatArray, wls: sax.FloatArray, phis: sax.FloatArray
) -> sax.FloatArray:
    """Grouped phase interpolation.

    .. note ::

        Grouped interpolation is useful to interpolate phase values where each datapoint
        is doubled (very close together) to give an indication of the phase
        variation at that point.

    .. warning ::

        this interpolation is only accurate in the range
        `[wls[0], wls[-2])` (`wls[-2]` not included). Any extrapolation
        outside these bounds can yield unexpected results!

    """
    wl = jnp.asarray(wl)
    wls = jnp.asarray(wls)
    phis = jnp.asarray(phis) % (2 * jnp.pi)
    phis = jnp.where(phis > jnp.pi, phis - 2 * jnp.pi, phis)
    if wls.ndim != 1:
        msg = "grouped_interp: wls should be a 1D array"
        raise ValueError(msg)
    if phis.ndim != 1:
        msg = "grouped_interp: wls should be a 1D array"
        raise ValueError(msg)
    if wls.shape != phis.shape:
        msg = "grouped_interp: wls and phis shape does not match"
        raise ValueError(msg)
    return _grouped_interp(wl.reshape(-1), wls, phis).reshape(*wl.shape)


def rename_params(model: sax.Model, renamings: dict[str, str]) -> sax.Model:
    """Rename the parameters of a `Model` or `ModelFactory`."""
    reversed_renamings = {v: k for k, v in renamings.items()}
    if len(reversed_renamings) < len(renamings):
        msg = "Multiple old names point to the same new name!"
        raise ValueError(msg)

    if (_model := sax.try_into[sax.Model](model)) is not None:

        @wraps(_model)
        def new_model(**settings: sax.SettingsValue) -> sax.SType:
            old_settings = {
                reversed_renamings.get(k, k): v for k, v in settings.items()
            }
            return _model(**old_settings)

        new_settings = {renamings.get(k, k): v for k, v in get_settings(_model).items()}
        _replace_kwargs(new_model, **new_settings)

        return cast(sax.Model, new_model)

    msg = "rename_params should be used to decorate a Model."
    raise ValueError(msg)


@overload
def rename_ports(S: sax.SDict, renamings: dict[str, str]) -> sax.SDict: ...


@overload
def rename_ports(S: sax.SCoo, renamings: dict[str, str]) -> sax.SCoo: ...


@overload
def rename_ports(S: sax.SDense, renamings: dict[str, str]) -> sax.SDense: ...


@overload
def rename_ports(S: sax.Model, renamings: dict[str, str]) -> sax.Model: ...


def rename_ports(
    S: sax.SType | sax.Model, renamings: dict[str, str]
) -> sax.SType | sax.Model:
    """Rename the ports of an `SDict`, `Model` or `ModelFactory`."""
    if (scoo := sax.try_into[sax.SCoo](S)) is not None:
        Si, Sj, Sx, ports_map = scoo
        ports_map = {renamings[p]: i for p, i in ports_map.items()}
        return Si, Sj, Sx, ports_map
    if (sdense := sax.try_into[sax.SDense](S)) is not None:
        Sx, ports_map = sdense
        ports_map = {renamings[p]: i for p, i in ports_map.items()}
        return Sx, ports_map
    if (sdict := sax.try_into[sax.SDict](S)) is not None:
        return {(renamings[p1], renamings[p2]): v for (p1, p2), v in sdict.items()}
    if (model := sax.try_into[sax.Model](S)) is not None:

        @wraps(model)
        def new_model(**settings: sax.SettingsValue) -> sax.SType:
            return rename_ports(model(**settings), renamings)

        return cast(sax.Model, new_model)
    msg = f"Cannot rename ports for type {type(S)}"
    raise ValueError(msg)


def hash_dict(dic: dict) -> int:
    """Hash a dictionary to an integer."""
    return int(
        md5(  # noqa: S324
            orjson.dumps(
                _numpyfy(dic), option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SORT_KEYS
            )
        ).hexdigest(),
        16,
    )


class Normalization(NamedTuple):
    """Normalization parameters for an array."""

    mean: sax.ComplexArray
    std: sax.ComplexArray


def normalization(x: sax.ComplexArray, axis: int | None = None) -> Normalization:
    """Calculate the mean and standard deviation of an array along a given axis."""
    if axis is None:
        return Normalization(x.mean(), x.std())
    return Normalization(x.mean(axis), x.std(axis))


def cartesian_product(*arrays: sax.ComplexArray) -> sax.ComplexArray:
    """Calculate the n-dimensional cartesian product the arrays."""
    ixarrays = jnp.ix_(*arrays)
    barrays = jnp.broadcast_arrays(*ixarrays)
    sarrays = jnp.stack(barrays, -1)
    product = sarrays.reshape(-1, sarrays.shape[-1])
    return product


def normalize(x: sax.ComplexArray, normalization: Normalization) -> sax.ComplexArray:
    """Normalize an array with a given mean and standard deviation."""
    mean, std = normalization
    return (x - mean) / std


def denormalize(x: sax.ComplexArray, normalization: Normalization) -> sax.ComplexArray:
    """Denormalize an array with a given mean and standard deviation."""
    mean, std = normalization
    return x * std + mean


def _generate_merged_dict(dict1: dict, dict2: dict) -> Iterator[tuple[Any, Any]]:
    # inspired by https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    keys = {
        **dict.fromkeys(dict1),
        **dict.fromkeys(dict2),
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


def _try_complex_float(f: Any) -> Any:  # noqa: ANN401
    """Try converting an object to float, return unchanged object on fail."""
    # TODO: deprecate for `sax.into` options.
    with warnings.catch_warnings():
        warnings.filterwarnings(action="error", category=ComplexWarning)
        try:
            return jnp.asarray(f, dtype=float)
        except ComplexWarning:
            return jnp.asarray(f, dtype=complex)
        except (ValueError, TypeError):
            pass

    return f


def _flatten_dict(
    dic: dict[str, Any], sep: str = ",", *, frozen: bool = False, parent_key: str = ""
) -> dict[str, Any]:
    items = []
    for k, v in dic.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(
                _flatten_dict(v, sep=sep, frozen=frozen, parent_key=new_key).items()
            )
        else:
            items.append((new_key, v))

    return dict(items)


@partial(jax.vmap, in_axes=(0, None, None), out_axes=0)
@jax.jit
def _grouped_interp(
    wl: float,  # 0D array (not-vmapped) ; 1D array (vmapped)
    wls: sax.FloatArray,  # 1D array
    phis: sax.FloatArray,  # 1D array
) -> sax.FloatArray:
    dphi_dwl = (phis[1::2] - phis[::2]) / (wls[1::2] - wls[::2])
    phis = phis[::2]
    wls = wls[::2]
    dwl = (wls[1:] - wls[:-1]).mean(0, keepdims=True)

    t = (wl - wls + 1e-5 * dwl) / dwl  # small offset to ensure no values are zero
    t = jnp.where(jnp.abs(t) < 1, t, 0)
    m0 = jnp.where(t > 0, size=1)[0]
    m1 = jnp.where(t < 0, size=1)[0]
    t = t[m0]
    wl0 = wls[m0]
    wl1 = wls[m1]
    phi0 = phis[m0]
    phi1 = phis[m1]
    dphi_dwl0 = dphi_dwl[m0]
    dphi_dwl1 = dphi_dwl[m1]
    _phi0 = phi0 - 0.5 * (wl1 - wl0) * (dphi_dwl0 * (t**2 - 2 * t) - dphi_dwl1 * t**2)
    _phi1 = phi1 - 0.5 * (wl1 - wl0) * (
        dphi_dwl0 * (t - 1) ** 2 - dphi_dwl1 * (t**2 - 1)
    )
    phis = jnp.arctan2(
        (1 - t) * jnp.sin(_phi0) + t * jnp.sin(_phi1),
        (1 - t) * jnp.cos(_phi0) + t * jnp.cos(_phi1),
    )
    return phis


def _replace_kwargs(func: Callable, **kwargs: Any) -> None:  # noqa: ANN401
    """Change the kwargs signature of a function."""
    sig = inspect.signature(func)
    settings = [
        inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=v)
        for k, v in kwargs.items()
    ]
    func.__signature__ = sig.replace(parameters=settings)  # type: ignore[reportFunctionMemberAccess]


def _numpyfy(obj: Any) -> Any:  # noqa: ANN401
    if not isinstance(obj, dict):
        return np.asarray(obj)
    return {k: _numpyfy(v) for k, v in obj.items()}
