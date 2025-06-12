"""SAX General Utilities."""

from __future__ import annotations

import inspect
import re
import warnings
from collections.abc import Callable, Iterable, Iterator
from functools import lru_cache, partial, wraps
from hashlib import md5
from typing import (
    Any,
    NamedTuple,
    cast,
    overload,
)

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.linalg
import numpy as np
import orjson
from natsort import natsorted
from numpy.exceptions import ComplexWarning

from .saxtypes import (
    ComplexArrayND,
    FloatArrayND,
    Model,
    ModelFactory,
    SCoo,
    SDense,
    SDict,
    Settings,
    SType,
    is_mixedmode,
    is_model,
    is_model_factory,
    is_scoo,
    is_sdense,
    is_sdict,
)


def block_diag(*arrs: ComplexArrayND) -> ComplexArrayND:
    """Create block diagonal matrix with arbitrary batch dimensions."""
    batch_shape = arrs[0].shape[:-2]

    N = 0
    for arr in arrs:
        if batch_shape != arr.shape[:-2]:
            msg = "batch dimensions for given arrays don't match."
            raise ValueError(msg)
        m, n = arr.shape[-2:]
        if m != n:
            msg = "given arrays are not square."
            raise ValueError(msg)
        N += n

    arrs = tuple(arr.reshape(-1, arr.shape[-2], arr.shape[-1]) for arr in arrs)
    batch_block_diag = jax.vmap(jsp.linalg.block_diag, in_axes=0, out_axes=0)
    block_diag = batch_block_diag(*arrs)
    return block_diag.reshape(*batch_shape, N, N)


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


def copy_settings(settings: Settings) -> Settings:
    """Copy a parameter dictionary."""
    return validate_settings(settings)  # validation also copies


def validate_settings(settings: Settings) -> Settings:
    """Validate a parameter dictionary."""
    _settings = {}
    for k, v in settings.items():
        if isinstance(v, dict):
            _settings[k] = validate_settings(v)
        else:
            _settings[k] = try_complex_float(v)
    return _settings


def try_complex_float(f: Any) -> Any:  # noqa: ANN401
    """Try converting an object to float, return unchanged object on fail."""
    with warnings.catch_warnings():
        warnings.filterwarnings(action="error", category=ComplexWarning)
        try:
            return jnp.asarray(f, dtype=float)
        except ComplexWarning:
            return jnp.asarray(f, dtype=complex)
        except (ValueError, TypeError):
            pass

    return f


def flatten_dict(dic: dict[str, Any], sep: str = ",") -> dict[str, Any]:
    """Flatten a nested dictionary."""
    return _flatten_dict(dic, sep=sep)


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


def get_ports(S: Model | SType) -> tuple[str, ...]:
    """Get port names of a model or an stype.

    .. note ::

        if a `Model` function is given in stead of an `SDict`,
        the function will be traced by JAX to obtain the port
        combinations of the resulting `SType`.
        Although this tracing of the function is 'cheap' in
        comparison to evaluating the model/circuit.
        It is not for free!  Use this function sparingly on your large
        `Model` or `circuit`!
    """
    if is_model(S):
        return _get_ports_from_model(cast(Model, S))
    if is_sdict(S):
        S = cast(SDict, S)
        ports_set = {p1 for p1, _ in S} | {p2 for _, p2 in S}
        return tuple(natsorted(ports_set))
    if is_scoo(S) or is_sdense(S):
        S = cast(SDense, S)
        *_, ports_map = S
        return tuple(natsorted(ports_map.keys()))
    msg = "Could not extract ports for given S"
    raise ValueError(msg)


@lru_cache(maxsize=4096)  # cache to prevent future tracing
def _get_ports_from_model(model: Model) -> tuple[str, ...]:
    # S: SType = jax.eval_shape(model)
    return get_ports(model())  # FIXME: this might be slow!


def get_port_combinations(S: Model | SType) -> tuple[tuple[str, str], ...]:
    """Get port combinations of a model or an stype."""
    if is_model(S):
        S = cast(Model, S)
        return _get_port_combinations_from_model(S)
    if is_sdict(S):
        S = cast(SDict, S)
        return tuple(S.keys())
    if is_scoo(S):
        Si, Sj, _, pm = cast(SCoo, S)
        rpm = {int(i): str(p) for p, i in pm.items()}
        return tuple(
            natsorted((rpm[int(i)], rpm[int(j)]) for i, j in zip(Si, Sj, strict=False))
        )
    if is_sdense(S):
        _, pm = cast(SDense, S)
        return tuple(natsorted((p1, p2) for p1 in pm for p2 in pm))
    msg = "Could not extract ports for given S"
    raise ValueError(msg)


@lru_cache(maxsize=4096)  # cache to prevent future tracing
def _get_port_combinations_from_model(model: Model) -> tuple[tuple[str, str], ...]:
    S: SType = jax.eval_shape(model)
    return get_port_combinations(S)


def get_settings(model: Model | ModelFactory) -> Settings:
    """Get the parameters of a SAX model function."""
    signature = inspect.signature(model)

    settings: Settings = {
        k: (v.default if not isinstance(v, dict) else v)
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    # make sure an inplace operation of resulting dict does not change the
    # circuit parameters themselves
    return copy_settings(settings)


def grouped_interp(
    wl: FloatArrayND, wls: FloatArrayND, phis: FloatArrayND
) -> FloatArrayND:
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

    @partial(jax.vmap, in_axes=(0, None, None), out_axes=0)
    @jax.jit
    def _grouped_interp(
        wl: float,  # 0D array (not-vmapped) ; 1D array (vmapped)
        wls: FloatArrayND,  # 1D array
        phis: FloatArrayND,  # 1D array
    ) -> FloatArrayND:
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
        _phi0 = phi0 - 0.5 * (wl1 - wl0) * (
            dphi_dwl0 * (t**2 - 2 * t) - dphi_dwl1 * t**2
        )
        _phi1 = phi1 - 0.5 * (wl1 - wl0) * (
            dphi_dwl0 * (t - 1) ** 2 - dphi_dwl1 * (t**2 - 1)
        )
        phis = jnp.arctan2(
            (1 - t) * jnp.sin(_phi0) + t * jnp.sin(_phi1),
            (1 - t) * jnp.cos(_phi0) + t * jnp.cos(_phi1),
        )
        return phis

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


def merge_dicts(*dicts: dict) -> dict:
    """Merge (possibly deeply nested) dictionaries."""
    if len(dicts) == 1:
        return dict(_generate_merged_dict(dicts[0], {}))
    if len(dicts) == 2:
        return dict(_generate_merged_dict(dicts[0], dicts[1]))
    return merge_dicts(dicts[0], merge_dicts(*dicts[1:]))


def _generate_merged_dict(dict1: dict, dict2: dict) -> Iterator[tuple[Any, Any]]:
    # inspired by https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    keys = dict.fromkeys(dict1) | dict.fromkeys(dict2)
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


def mode_combinations(
    modes: Iterable[str], *, cross: bool = False
) -> tuple[tuple[str, str], ...]:
    """Create mode combinations for a collection of given modes."""
    if cross:
        mode_combinations = natsorted((m1, m2) for m1 in modes for m2 in modes)
    else:
        mode_combinations = natsorted((m, m) for m in modes)
    return tuple(mode_combinations)


def reciprocal(sdict: SDict) -> SDict:
    """Make an SDict reciprocal."""
    if is_sdict(sdict):
        return {
            **{(p1, p2): v for (p1, p2), v in sdict.items()},
            **{(p2, p1): v for (p1, p2), v in sdict.items()},
        }
    msg = "sax.reciprocal is only valid for SDict types"
    raise ValueError(msg)


@overload
def rename_params(model: ModelFactory, renamings: dict[str, str]) -> ModelFactory: ...


@overload
def rename_params(model: Model, renamings: dict[str, str]) -> Model: ...


def rename_params(
    model: Model | ModelFactory, renamings: dict[str, str]
) -> Model | ModelFactory:
    """Rename the parameters of a `Model` or `ModelFactory`."""
    reversed_renamings = {v: k for k, v in renamings.items()}
    if len(reversed_renamings) < len(renamings):
        msg = "Multiple old names point to the same new name!"
        raise ValueError(msg)

    if is_model_factory(model):
        old_model_factory = cast(ModelFactory, model)
        old_settings = get_settings(model)

        @wraps(old_model_factory)
        def new_model_factory(**settings: Any) -> Model:  # noqa: ANN401
            old_settings = {
                reversed_renamings.get(k, k): v for k, v in settings.items()
            }
            model = old_model_factory(**old_settings)
            return rename_params(model, renamings)

        new_settings = {renamings.get(k, k): v for k, v in old_settings.items()}
        _replace_kwargs(new_model_factory, **new_settings)

        return new_model_factory

    if is_model(model):
        old_model = cast(Model, model)
        old_settings = get_settings(model)

        @wraps(old_model)
        def new_model(**settings: Any) -> SType:  # noqa: ANN401
            old_settings = {
                reversed_renamings.get(k, k): v for k, v in settings.items()
            }
            return old_model(**old_settings)

        new_settings = {renamings.get(k, k): v for k, v in old_settings.items()}
        _replace_kwargs(new_model, **new_settings)

        return new_model

    msg = "rename_params should be used to decorate a Model or ModelFactory."
    raise ValueError(msg)


def _replace_kwargs(func: Callable, **kwargs: Any) -> None:  # noqa: ANN401
    """Change the kwargs signature of a function."""
    sig = inspect.signature(func)
    settings = [
        inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=v)
        for k, v in kwargs.items()
    ]
    func.__signature__ = sig.replace(parameters=settings)  # type: ignore[reportFunctionMemberAccess]


@overload
def rename_ports(S: SDict, renamings: dict[str, str]) -> SDict: ...


@overload
def rename_ports(S: SCoo, renamings: dict[str, str]) -> SCoo: ...


@overload
def rename_ports(S: SDense, renamings: dict[str, str]) -> SDense: ...


@overload
def rename_ports(S: Model, renamings: dict[str, str]) -> Model: ...


@overload
def rename_ports(S: ModelFactory, renamings: dict[str, str]) -> ModelFactory: ...


def rename_ports(
    S: SType | Model | ModelFactory, renamings: dict[str, str]
) -> SType | Model | ModelFactory:
    """Rename the ports of an `SDict`, `Model` or `ModelFactory`."""
    if is_scoo(S):
        Si, Sj, Sx, ports_map = cast(SCoo, S)
        ports_map = {renamings[p]: i for p, i in ports_map.items()}
        return Si, Sj, Sx, ports_map
    if is_sdense(S):
        Sx, ports_map = cast(SDense, S)
        ports_map = {renamings[p]: i for p, i in ports_map.items()}
        return Sx, ports_map
    if is_sdict(S):
        sdict = cast(SDict, S)
        original_ports = get_ports(sdict)
        if len(renamings) != len(original_ports):
            msg = "Port renamings caused port duplication!"
            raise ValueError(msg)
        return {(renamings[p1], renamings[p2]): v for (p1, p2), v in sdict.items()}
    if is_model(S):
        old_model = cast(Model, S)

        @wraps(old_model)
        def new_model(**settings: Any) -> SType:  # noqa: ANN401
            return rename_ports(old_model(**settings), renamings)

        return new_model
    if is_model_factory(S):
        old_model_factory = cast(ModelFactory, S)

        @wraps(old_model_factory)
        def new_model_factory(**settings: Any) -> Callable[..., SType]:  # noqa: ANN401
            return rename_ports(old_model_factory(**settings), renamings)

        return new_model_factory
    msg = f"Cannot rename ports for type {type(S)}"
    raise ValueError(msg)


def update_settings(settings: Settings, *compnames: str, **kwargs: Any) -> Settings:  # noqa: ANN401
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
                _settings[k] = try_complex_float(kwargs[k])
            else:
                _settings[k] = try_complex_float(v)
    else:
        for k, v in settings.items():
            if isinstance(v, dict):
                if k == compnames[0]:
                    _settings[k] = update_settings(v, *compnames[1:], **kwargs)
                else:
                    _settings[k] = v
            else:
                _settings[k] = try_complex_float(v)
    return _settings


def validate_not_mixedmode(S: SType) -> None:
    """Validate that an stype is not 'mixed mode' (i.e. invalid).

    Args:
        S: the stype to validate
    """
    if is_mixedmode(S):  # mixed mode
        msg = (
            "Given SType is neither multimode or singlemode. Please check the port "
            "names: they should either ALL contain the '@' separator (multimode) "
            "or NONE should contain the '@' separator (singlemode)."
        )
        raise ValueError(msg)


def validate_multimode(S: SType, modes: tuple[str, ...] = ("te", "tm")) -> None:
    """Validate that an stype is multimode and that the given modes are present."""
    try:
        current_modes = {p.split("@")[1] for p in get_ports(S)}
    except IndexError as err:
        msg = "The given stype is not multimode."
        raise ValueError(msg) from err
    for mode in modes:
        if mode not in current_modes:
            msg = f"Could not find mode '{mode}' in one of the multimode models."
            raise ValueError(msg)


def validate_sdict(sdict: Any) -> None:  # noqa: ANN401
    """Validate an `SDict`."""
    if not isinstance(sdict, dict):
        msg = "An SDict should be a dictionary."
        raise TypeError(msg)
    for ports in sdict:
        if not isinstance(ports, tuple) and len(ports) != 2:
            msg = f"SDict keys should be length-2 tuples. Got {ports}"
            raise TypeError(msg)
        p1, p2 = ports
        if not isinstance(p1, str) or not isinstance(p2, str):
            msg = (
                f"SDict ports should be strings. Got {ports} "
                f"({type(ports[0])}, {type(ports[1])})"
            )
            raise TypeError(msg)


def get_inputs_outputs(
    ports: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Get input and output ports from a tuple of port names."""
    inputs = tuple(p for p in ports if p.lower().startswith("in"))
    outputs = tuple(p for p in ports if not p.lower().startswith("in"))
    if not inputs:
        inputs = tuple(p for p in ports if not p.lower().startswith("out"))
        outputs = tuple(p for p in ports if p.lower().startswith("out"))
    return inputs, outputs


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


def _numpyfy(obj: Any) -> Any:  # noqa: ANN401
    if not isinstance(obj, dict):
        return np.asarray(obj)
    return {k: _numpyfy(v) for k, v in obj.items()}


class Normalization(NamedTuple):
    """Normalization parameters for an array."""

    mean: ComplexArrayND
    std: ComplexArrayND


def normalization(x: ComplexArrayND, axis: int | None = None) -> Normalization:
    """Calculate the mean and standard deviation of an array along a given axis."""
    if axis is None:
        return Normalization(x.mean(), x.std())
    return Normalization(x.mean(axis), x.std(axis))


def cartesian_product(*arrays: ComplexArrayND) -> ComplexArrayND:
    """Calculate the n-dimensional cartesian product the arrays."""
    ixarrays = jnp.ix_(*arrays)
    barrays = jnp.broadcast_arrays(*ixarrays)
    sarrays = jnp.stack(barrays, -1)
    product = sarrays.reshape(-1, sarrays.shape[-1])
    return product


def normalize(x: ComplexArrayND, normalization: Normalization) -> ComplexArrayND:
    """Normalize an array with a given mean and standard deviation."""
    mean, std = normalization
    return (x - mean) / std


def denormalize(x: ComplexArrayND, normalization: Normalization) -> ComplexArrayND:
    """Denormalize an array with a given mean and standard deviation."""
    mean, std = normalization
    return x * std + mean
