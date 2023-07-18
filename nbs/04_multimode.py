# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# default_exp multimode
# -

# # Multimode
#
# > SAX Multimode utils

# +
# hide
from fastcore.test import test_eq

import os, sys; sys.stderr = open(os.devnull, "w")

# +
# export
from __future__ import annotations

from functools import wraps
from typing import Dict, Tuple, Union, cast, overload

from sax.typing_ import (
    Model,
    SCoo,
    SDense,
    SDict,
    SType,
    is_model,
    is_multimode,
    is_scoo,
    is_sdense,
    is_sdict,
    is_singlemode,
    _consolidate_sdense,
)
from sax.utils import (
    block_diag,
    mode_combinations,
    validate_multimode,
    validate_not_mixedmode,
)

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False


# +
# exporti

@overload
def multimode(S: Model, modes: Tuple[str, ...] = ("te", "tm")) -> Model:
    ...


@overload
def multimode(S: SDict, modes: Tuple[str, ...] = ("te", "tm")) -> SDict:
    ...


@overload
def multimode(S: SCoo, modes: Tuple[str, ...] = ("te", "tm")) -> SCoo:
    ...


@overload
def multimode(S: SDense, modes: Tuple[str, ...] = ("te", "tm")) -> SDense:
    ...


# +
# export

def multimode(
    S: Union[SType, Model], modes: Tuple[str, ...] = ("te", "tm")
) -> Union[SType, Model]:
    """Convert a single mode model to a multimode model"""
    if is_model(S):
        model = cast(Model, S)

        @wraps(model)
        def new_model(**params):
            return multimode(model(**params), modes=modes)

        return cast(Model, new_model)

    S = cast(SType, S)

    validate_not_mixedmode(S)
    if is_multimode(S):
        validate_multimode(S, modes=modes)
        return S

    if is_sdict(S):
        return _multimode_sdict(cast(SDict, S), modes=modes)
    elif is_scoo(S):
        return _multimode_scoo(cast(SCoo, S), modes=modes)
    elif is_sdense(S):
        return _multimode_sdense(cast(SDense, S), modes=modes)
    else:
        raise ValueError("cannot convert to multimode. Unknown stype.")


def _multimode_sdict(sdict: SDict, modes: Tuple[str, ...] = ("te", "tm")) -> SDict:
    multimode_sdict = {}
    _mode_combinations = mode_combinations(modes)
    for (p1, p2), value in sdict.items():
        for (m1, m2) in _mode_combinations:
            multimode_sdict[f"{p1}@{m1}", f"{p2}@{m2}"] = value
    return multimode_sdict


def _multimode_scoo(scoo: SCoo, modes: Tuple[str, ...] = ("te", "tm")) -> SCoo:

    Si, Sj, Sx, port_map = scoo
    num_ports = len(port_map)
    mode_map = (
        {mode: i for i, mode in enumerate(modes)}
        if not isinstance(modes, dict)
        else cast(Dict, modes)
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


def _multimode_sdense(sdense, modes=("te", "tm")):

    Sx, port_map = sdense
    num_ports = len(port_map)
    mode_map = (
        {mode: i for i, mode in enumerate(modes)}
        if not isinstance(modes, dict)
        else modes
    )

    Sx_m = block_diag(*(Sx for _ in modes))

    port_map_m = {
        f"{port}@{mode}": idx + mode_map[mode] * num_ports
        for mode in modes
        for port, idx in port_map.items()
    }

    return Sx_m, port_map_m


# -

sdict_s = {("in0", "out0"): 1.0}
sdict_m = multimode(sdict_s)
assert sdict_m == {("in0@te", "out0@te"): 1.0, ("in0@tm", "out0@tm"): 1.0}

from sax.typing_ import scoo
scoo_s = scoo(sdict_s)
scoo_m = multimode(scoo_s)
test_eq(scoo_m[0], jnp.array([0, 2], dtype=int))
test_eq(scoo_m[1], jnp.array([1, 3], dtype=int))
test_eq(scoo_m[2], jnp.array([1.0, 1.0], dtype=float))
test_eq(scoo_m[3], {"in0@te": 0, "out0@te": 1, "in0@tm": 2, "out0@tm": 3})

from sax.typing_ import sdense
sdense_s = sdense(sdict_s)
sdense_m = multimode(sdense_s)
test_eq(
    sdense_m[0],
    [[0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
     [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
     [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
     [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]],
)
test_eq(sdense_m[1], {"in0@te": 0, "out0@te": 1, "in0@tm": 2, "out0@tm": 3})


# +
# exporti

@overload
def singlemode(S: Model, mode: str = "te") -> Model:
    ...


@overload
def singlemode(S: SDict, mode: str = "te") -> SDict:
    ...


@overload
def singlemode(S: SCoo, mode: str = "te") -> SCoo:
    ...


@overload
def singlemode(S: SDense, mode: str = "te") -> SDense:
    ...


# +
# export

def singlemode(S: Union[SType, Model], mode: str = "te") -> Union[SType, Model]:
    """Convert multimode model to a singlemode model"""
    if is_model(S):
        model = cast(Model, S)

        @wraps(model)
        def new_model(**params):
            return singlemode(model(**params), mode=mode)

        return cast(Model, new_model)

    S = cast(SType, S)

    validate_not_mixedmode(S)
    if is_singlemode(S):
        return S
    if is_sdict(S):
        return _singlemode_sdict(cast(SDict, S), mode=mode)
    elif is_scoo(S):
        return _singlemode_scoo(cast(SCoo, S), mode=mode)
    elif is_sdense(S):
        return _singlemode_sdense(cast(SDense, S), mode=mode)
    else:
        raise ValueError("cannot convert to multimode. Unknown stype.")


def _singlemode_sdict(sdict: SDict, mode: str = "te") -> SDict:
    singlemode_sdict = {}
    for (p1, p2), value in sdict.items():
        if p1.endswith(f"@{mode}") and p2.endswith(f"@{mode}"):
            p1, _ = p1.split("@")
            p2, _ = p2.split("@")
            singlemode_sdict[p1, p2] = value
    return singlemode_sdict


def _singlemode_scoo(scoo: SCoo, mode: str = "te") -> SCoo:
    Si, Sj, Sx, port_map = scoo
    # no need to touch the data...
    # just removing some ports from the port map should be enough
    port_map = {
        port.split("@")[0]: idx
        for port, idx in port_map.items()
        if port.endswith(f"@{mode}")
    }
    return Si, Sj, Sx, port_map


def _singlemode_sdense(sdense: SDense, mode: str = "te") -> SDense:
    Sx, port_map = sdense
    # no need to touch the data...
    # just removing some ports from the port map should be enough
    port_map = {
        port.split("@")[0]: idx
        for port, idx in port_map.items()
        if port.endswith(f"@{mode}")
    }
    return _consolidate_sdense(Sx, port_map)


# -

sdict_s = singlemode(sdict_m)
assert sdict_s == {("in0", "out0"): 1.0}

scoo_s = singlemode(scoo_s)
test_eq(scoo_s[0], jnp.array([0], dtype=int))
test_eq(scoo_s[1], jnp.array([1], dtype=int))
test_eq(scoo_s[2], jnp.array([1.0], dtype=float))
test_eq(scoo_s[3], {'in0': 0, 'out0': 1})

sdense_s = singlemode(sdense_m)
test_eq(
    sdense_s[0],
    [[0.0 + 0.0j, 1.0 + 0.0j],
     [0.0 + 0.0j, 0.0 + 0.0j]]
)
test_eq(sdense_s[1], {'in0': 0, 'out0': 1})
