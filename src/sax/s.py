"""SAX S-Matrix utilities."""

from __future__ import annotations

from functools import wraps
from typing import cast, overload

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array
from natsort import natsorted
from pydantic import validate_call

import sax

from .multimode import _consolidate_sdense

__all__ = [
    "block_diag",
    "get_mode",
    "get_modes",
    "get_ports",
    "reciprocal",
    "scoo",
    "sdense",
    "sdict",
]


@overload
def sdict(S: sax.Model) -> sax.SDictModel: ...


@overload
def sdict(S: sax.SType) -> sax.SDict: ...


def sdict(S: sax.Model | sax.SType) -> sax.SDictModel | sax.SDict:
    """Convert an `SCoo` or `SDense` to `SDict`."""
    if (_model := sax.try_into[sax.Model](S)) is not None:

        @wraps(_model)
        def model(**kwargs: sax.SettingsValue) -> sax.Model:
            return sdict(_model(**kwargs))

        return cast(sax.SDictModel, model)

    if (_scoo := sax.try_into[sax.SCoo](S)) is not None:
        return _scoo_to_sdict(*_scoo)

    if (_sdense := sax.try_into[sax.SDense](S)) is not None:
        return _sdense_to_sdict(*_sdense)

    if (_sdict := sax.try_into[sax.SDict](S)) is not None:
        return _sdict

    msg = f"Could not convert S-matrix to sdict. Got: {S!r}."
    raise ValueError(msg)


@overload
def scoo(S: sax.Model) -> sax.SCooModel: ...


@overload
def scoo(S: sax.SType) -> sax.SCoo: ...


def scoo(S: sax.Model | sax.SType) -> sax.SCooModel | sax.SCoo:
    """Convert an `SDict` or `SDense` to `SCoo`."""
    if (_model := sax.try_into[sax.Model](S)) is not None:

        @wraps(_model)
        def model(**kwargs: sax.SettingsValue) -> sax.SCoo:
            return scoo(_model(**kwargs))

        return model

    if (_scoo := sax.try_into[sax.SCoo](S)) is not None:
        return _scoo

    if (_sdense := sax.try_into[sax.SDense](S)) is not None:
        return _sdense_to_scoo(*_sdense)

    if (_sdict := sax.try_into[sax.SDict](S)) is not None:
        return _sdict_to_scoo(_sdict)

    msg = f"Could not convert S-matrix to scoo. Got: {S!r}."
    raise ValueError(msg)


@overload
def sdense(S: sax.Model) -> sax.SDenseModel: ...


@overload
def sdense(S: sax.SType) -> sax.SDense: ...


def sdense(S: sax.SType | sax.Model) -> sax.SDenseModel | sax.SDense:
    """Convert an `SDict` or `SCoo` to `SDense`."""
    if (_model := sax.try_into[sax.Model](S)) is not None:

        @wraps(_model)
        def model(**kwargs: sax.SettingsValue) -> sax.SDense:
            return sdense(_model(**kwargs))

        return model

    if (_scoo := sax.try_into[sax.SCoo](S)) is not None:
        return _scoo_to_sdense(*_scoo)

    if (_sdense := sax.try_into[sax.SDense](S)) is not None:
        return _sdense

    if (_sdict := sax.try_into[sax.SDict](S)) is not None:
        return _sdict_to_sdense(*_sdict)

    msg = f"Could not convert S-matrix to sdense. Got: {S!r}."
    raise ValueError(msg)


def reciprocal(sdict: sax.SDict) -> sax.SDict:
    """Make an SDict reciprocal."""
    sdict = sax.into[sax.SDict](sdict)
    return {
        **{(p1, p2): v for (p1, p2), v in sdict.items()},
        **{(p2, p1): v for (p1, p2), v in sdict.items()},
    }


def block_diag(*arrs: Array) -> Array:
    """Create block diagonal matrix with arbitrary batch dimensions."""
    batch_shape = arrs[0].shape[:-2]

    N = 0
    for arr in arrs:
        if batch_shape != arr.shape[:-2]:
            msg = "Batch dimensions for given arrays don't match."
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


def get_ports(S: sax.SType) -> tuple[sax.Port, ...] | tuple[sax.PortMode]:
    """Get port names of a model or an stype."""
    if callable(S):
        msg = (
            "Getting the ports of a model is no longer supported. "
            "Please Evaluate the model first: Use get_ports(model()) in stead of "
            f"get_ports(model). Got: {S}"
        )
        raise TypeError(msg)
    if (sdict := sax.try_into[sax.SDict](S)) is not None:
        ports_set = {p1 for p1, _ in sdict} | {p2 for _, p2 in sdict}
        return tuple(natsorted(ports_set))
    if (with_pm := sax.try_into[sax.SCoo | sax.SDense](S)) is not None:
        *_, pm = with_pm
        return tuple(natsorted(pm.keys()))

    msg = f"Expected an SType. Got: {S!r} [{type(S)}]"
    raise TypeError(msg)


@validate_call
def get_modes(S: sax.STypeMM) -> tuple[sax.Mode, ...]:
    """Get the modes in a multimode S-matrix."""
    return tuple(get_mode(pm) for pm in get_ports(S))


@validate_call
def get_mode(pm: sax.PortMode) -> sax.Mode:
    """Get the mode from a port@mode string."""
    return pm.split("@")[1]


def _scoo_to_sdict(
    Si: sax.IntArray1D,
    Sj: sax.IntArray1D,
    Sx: sax.ComplexArray,
    ports_map: sax.PortMap,
) -> sax.SDict:
    sdict = {}
    inverse_ports_map = {int(i): p for p, i in ports_map.items()}
    for i, (si, sj) in enumerate(zip(Si, Sj, strict=True)):
        input_port = inverse_ports_map.get(int(si), "")
        output_port = inverse_ports_map.get(int(sj), "")
        sdict[input_port, output_port] = Sx[..., i]
    return {(p1, p2): v for (p1, p2), v in sdict.items() if p1 and p2}


def _sdense_to_sdict(S: Array, ports_map: sax.PortMap) -> sax.SDict:
    sdict = {}
    for p1, i in ports_map.items():
        for p2, j in ports_map.items():
            sdict[p1, p2] = S[..., i, j]
    return sdict


def _sdict_to_scoo(sdict: sax.SDict) -> sax.SCoo:
    all_ports = {}
    for p1, p2 in sdict:
        all_ports[p1] = None
        all_ports[p2] = None
    ports_map = {p: int(i) for i, p in enumerate(all_ports)}
    Sx = jnp.stack(jnp.broadcast_arrays(*sdict.values()), -1)
    Si = jnp.array([ports_map[p] for p, _ in sdict])
    Sj = jnp.array([ports_map[p] for _, p in sdict])
    return Si, Sj, Sx, ports_map


def _sdense_to_scoo(S: sax.ComplexArray, ports_map: sax.PortMap) -> sax.SCoo:
    S, ports_map = _consolidate_sdense((S, ports_map))
    Sj, Si = jnp.meshgrid(jnp.arange(S.shape[-1]), jnp.arange(S.shape[-2]))
    return Si.ravel(), Sj.ravel(), S.reshape(*S.shape[:-2], -1), ports_map


def _scoo_to_sdense(
    Si: sax.IntArray1D,
    Sj: sax.IntArray1D,
    Sx: sax.ComplexArray,
    ports_map: dict[str, int],
) -> sax.SDense:
    n_col = len(ports_map)
    S = jnp.zeros((*Sx.shape[:-1], n_col, n_col), dtype=complex)
    S = S.at[..., Si, Sj].add(Sx)
    return S, ports_map


def _sdict_to_sdense(sdict: sax.SDict) -> sax.SDense:
    Si, Sj, Sx, ports_map = _sdict_to_scoo(sdict)
    return _scoo_to_sdense(Si, Sj, Sx, ports_map)
