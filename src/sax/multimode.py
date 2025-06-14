"""SAX Multimode support."""

from __future__ import annotations

from functools import wraps
from typing import Any, cast, overload

import jax.numpy as jnp

from .saxtypes import (
    Model,
    SCoo,
    SDense,
    SDict,
    SType,
    _consolidate_sdense,
    is_model,
    is_multimode,
    is_scoo,
    is_sdense,
    is_sdict,
    is_singlemode,
)
from .utils import (
    block_diag,
    mode_combinations,
    validate_multimode,
    validate_not_mixedmode,
)


@overload
def multimode(S: Model, modes: tuple[str, ...] = ("TE", "TM")) -> Model: ...


@overload
def multimode(S: SDict, modes: tuple[str, ...] = ("TE", "TM")) -> SDict: ...


@overload
def multimode(S: SCoo, modes: tuple[str, ...] = ("TE", "TM")) -> SCoo: ...


@overload
def multimode(S: SDense, modes: tuple[str, ...] = ("TE", "TM")) -> SDense: ...


def multimode(S: SType | Model, modes: tuple[str, ...] = ("TE", "TM")) -> SType | Model:
    """Convert a single mode model to a multimode model."""
    if is_model(S):
        model = cast(Model, S)

        @wraps(model)
        def new_model(**params: Any) -> SType:  # noqa: ANN401
            return multimode(model(**params), modes=modes)

        return cast(Model, new_model)

    S = cast(SType, S)

    validate_not_mixedmode(S)
    if is_multimode(S):
        validate_multimode(S, modes=modes)
        return S

    if is_sdict(S):
        return _multimode_sdict(cast(SDict, S), modes=modes)
    if is_scoo(S):
        return _multimode_scoo(cast(SCoo, S), modes=modes)
    if is_sdense(S):
        return _multimode_sdense(cast(SDense, S), modes=modes)

    msg = "cannot convert to multimode. Unknown SType."
    raise ValueError(msg)


def _multimode_sdict(sdict: SDict, modes: tuple[str, ...] = ("TE", "TM")) -> SDict:
    multimode_sdict = {}
    _mode_combinations = mode_combinations(modes)
    for (p1, p2), value in sdict.items():
        for m1, m2 in _mode_combinations:
            multimode_sdict[f"{p1}@{m1}", f"{p2}@{m2}"] = value
    return multimode_sdict


def _multimode_scoo(scoo: SCoo, modes: tuple[str, ...] = ("TE", "TM")) -> SCoo:
    Si, Sj, Sx, port_map = scoo
    num_ports = len(port_map)
    mode_map = (
        {mode: i for i, mode in enumerate(modes)}
        if not isinstance(modes, dict)
        else cast(dict, modes)
    )

    _mode_combinations = mode_combinations(modes)

    Si_m = jnp.concatenate(
        [Si + mode_map[m] * num_ports for m, _ in _mode_combinations], -1
    )
    Sj_m = jnp.concatenate(
        [Sj + mode_map[m] * num_ports for _, m in _mode_combinations], -1
    )
    Sx_m = jnp.concatenate([Sx for _ in _mode_combinations], -1)
    port_map_m = {
        f"{port}@{mode}": idx + mode_map[mode] * num_ports
        for mode in modes
        for port, idx in port_map.items()
    }

    return Si_m, Sj_m, Sx_m, port_map_m


def _multimode_sdense(sdense: SDense, modes: tuple[str, ...] = ("TE", "TM")) -> SDense:
    Sx, port_map = sdense
    num_ports = len(port_map)
    mode_map: dict[str, int] = (
        {mode: i for i, mode in enumerate(modes)}
        if not isinstance(modes, dict)
        else modes
    )

    Sx_m = block_diag(*(Sx for _ in modes))

    port_map_m = {
        f"{port}@{mode}": idx + mode_map[mode] * num_ports  # type: ignore[reportCallIssue]
        for mode in modes
        for port, idx in port_map.items()
    }

    return Sx_m, port_map_m


@overload
def singlemode(S: Model, mode: str = "TE") -> Model: ...


@overload
def singlemode(S: SDict, mode: str = "TE") -> SDict: ...


@overload
def singlemode(S: SCoo, mode: str = "TE") -> SCoo: ...


@overload
def singlemode(S: SDense, mode: str = "TE") -> SDense: ...


def singlemode(S: SType | Model, mode: str = "TE") -> SType | Model:
    """Convert multimode model to a singlemode model."""
    if is_model(S):
        model = cast(Model, S)

        @wraps(model)
        def new_model(**params: Any) -> SType:  # noqa: ANN401
            return singlemode(model(**params), mode=mode)

        return cast(Model, new_model)

    S = cast(SType, S)

    validate_not_mixedmode(S)
    if is_singlemode(S):
        return S
    if is_sdict(S):
        return _singlemode_sdict(cast(SDict, S), mode=mode)
    if is_scoo(S):
        return _singlemode_scoo(cast(SCoo, S), mode=mode)
    if is_sdense(S):
        return _singlemode_sdense(cast(SDense, S), mode=mode)

    msg = "cannot convert to multimode. Unknown SType."
    raise ValueError(msg)


def _singlemode_sdict(sdict: SDict, mode: str = "TE") -> SDict:
    singlemode_sdict = {}
    for (p1, p2), value in sdict.items():
        if p1.endswith(f"@{mode}") and p2.endswith(f"@{mode}"):
            p1, _ = p1.split("@")
            p2, _ = p2.split("@")
            singlemode_sdict[p1, p2] = value
    return singlemode_sdict


def _singlemode_scoo(scoo: SCoo, mode: str = "TE") -> SCoo:
    Si, Sj, Sx, port_map = scoo
    # no need to touch the data...
    # just removing some ports from the port map should be enough
    port_map = {
        port.split("@")[0]: idx
        for port, idx in port_map.items()
        if port.endswith(f"@{mode}")
    }
    return Si, Sj, Sx, port_map


def _singlemode_sdense(sdense: SDense, mode: str = "TE") -> SDense:
    Sx, port_map = sdense
    # no need to touch the data...
    # just removing some ports from the port map should be enough
    port_map = {
        port.split("@")[0]: idx
        for port, idx in port_map.items()
        if port.endswith(f"@{mode}")
    }
    return _consolidate_sdense(Sx, port_map)
