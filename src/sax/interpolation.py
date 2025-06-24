"""Multi-dimensional interpolation utilitites."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from jaxtyping import Array
from numpy.typing import NDArray

import sax

__all__ = [
    "interpolate_xarray",
    "to_df",
    "to_xarray",
]


def interpolate_xarray(
    xarr: xr.DataArray,
    /,
    **kwargs: sax.FloatArray,
) -> dict[str, sax.FloatArray]:
    """Interpolate a multi-dimensional xarray with JAX.

    Args:
        xarr: the xarray to do a grid-interpolation over
        **kwargs: the other parameters to interpolate over
    """
    with jax.ensure_compile_time_eval():
        data = jnp.asarray(xarr)

        # don't use jnp.asarray here as values can be strings!
        params = {k: np.asarray(xarr[k]) for k in xarr.dims}

        strings: dict[str, dict[str, int]] = kwargs.pop("strings", {})  # type: ignore[reportAssignmentType]
        target_name = xarr.dims[-1]
        params.pop(target_name, None)
        strings.pop(target_name, None)  # type: ignore[reportArgumentType]

        # don't use jnp.asarray here as values can be strings!
        targets = {
            str(k): i for i, k in enumerate(np.asarray(xarr.coords[target_name]))
        }

        params["targets"] = np.arange(0, len(targets), 1, dtype=np.uint8)
        strings = {**strings, "targets": targets}

    s, axs, pos = _evaluate_general_corner_model(
        data,
        params,  # type: ignore[reportArgumentType]
        strings,
        **kwargs,
    )
    return {k: s.take(pos["targets"][k], axs["targets"]) for k in targets}


def to_xarray(
    stacked_data: pd.DataFrame,
    *,
    target_names: Iterable[str] = ("amp", "phi"),
) -> xr.DataArray:
    """Converts a stacked dataframe into a multi-dimensional xarray.

    Args:
        stacked_data: the data to convert
        target_names: the names of the target columns.
    """
    df = stacked_data.copy()
    for name in target_names:
        if name not in df.columns:
            msg = (
                f"target name {name!r} not found in dataframe. "
                f"Got a dataframe with columns: {df.columns}."
            )
            raise ValueError(msg)
    param_names = [c for c in df.columns if c not in target_names]
    df = cast(pd.DataFrame, df[[*param_names, *target_names]])
    df = df.sort_values(by=param_names).reset_index(drop=True)
    params: dict[str, Array | NDArray] = {
        c: np.asarray(df[c].unique()) for c in param_names
    }
    params["targets"] = np.array(list(target_names), dtype=object)
    data = df[target_names].to_numpy()
    data = data.reshape(*(v.shape[0] for v in params.values()))
    data = jnp.asarray(data)
    coords, strings = _extract_strings(params)
    coords["targets"] = np.asarray(list(strings.pop("targets")), dtype=object)  # type: ignore[reportArgumentType]
    if strings:
        msg = f"Found non-float columns in the dataframe: {strings}."
        raise ValueError(msg)
    return xr.DataArray(data=data, coords=coords)


def to_df(xarr: xr.DataArray, *, target_name: str = "target") -> pd.DataFrame:
    """Converts a multi-dimensional xarray into a stacked dataframe."""
    stacked = xarr.stack(__stacked_dim=xarr.dims)  # noqa: PD013
    df = (
        stacked.reset_index("__stacked_dim")
        .to_dataframe(name=target_name)
        .reset_index(drop=True)
    )
    return df


def _evaluate_general_corner_model(
    data: Array,
    params: dict[str, Array],
    strings: dict[str, dict[str, int]],
    /,
    **kwargs: Array,
) -> tuple[Array, dict[str, int], dict[str, dict[str, int]]]:
    """Hypercube representation interpolator.

    Args:
        data: the data to interpolate (output of to_hypercube)
        params: the parameter sample points (output of to_hypercube)
        strings: non-interpolatable parameters (like strings)
            with their values (output of to_hypercube)
        **kwargs: the values at which to interpolate the s-parameter.
            These values should be broadcastable into each other.

    Returns:
        the interpolated S-matrix at the values of **kwargs
    """
    given_params = {k: kwargs.get(k, jnp.asarray(v).mean()) for k, v in params.items()}
    given_strings = {
        p: (list(v.values()) if p not in kwargs else [v[vv] for vv in kwargs[p]])
        for p, v in strings.items()
    }
    string_locations = {k: i for i, k in enumerate(given_params) if k in given_strings}
    param_locations = {
        k: i for i, k in enumerate(given_params) if k not in given_strings
    }
    data = jnp.transpose(
        data, axes=[*param_locations.values(), *string_locations.values()]
    )
    given_params = {k: v for k, v in given_params.items() if k not in given_strings}
    num_params = len(given_params)
    data = _downselect(data, list(given_strings.values()))
    param_shape, string_shape = data.shape[:num_params], data.shape[num_params:]
    data = data.reshape(*param_shape, -1)
    given_param_values = [jnp.asarray(v) for v in given_params.values()]
    stacked_params = jnp.stack(jnp.broadcast_arrays(*given_param_values), 0)
    coords = _get_coordinates(
        [v for k, v in params.items() if k in given_params], stacked_params
    )
    result = _map_coordinates(data, coords)
    axs = {k: i for i, k in enumerate(given_strings)}
    rev_strings = {k: {vv: kk for kk, vv in v.items()} for k, v in strings.items()}
    pos = {
        k: {rev_strings[k][vv]: i for i, vv in enumerate(v)}
        for k, v in given_strings.items()
    }
    return result.reshape(*string_shape, *result.shape[1:]), axs, pos


def _downselect(data: Array, idxs_list: list[list[int]]) -> Array:
    for i, idxs in enumerate(reversed(idxs_list), start=1):
        data = data.take(np.asarray(idxs, dtype=int), axis=-i)
    return data


@partial(jax.vmap, in_axes=(-1, None), out_axes=0)
def _map_coordinates(input: Array, coordinates: Sequence[Array]) -> Array:  # noqa: A002
    return jax.scipy.ndimage.map_coordinates(input, coordinates, 1, mode="nearest")


def _get_coordinate(arr1d: Array, value: Array) -> Array:
    return jnp.interp(value, arr1d, jnp.arange(arr1d.shape[0]))


def _get_coordinates(arrs1d: Sequence[Array], values: jnp.ndarray) -> list[Array]:
    # don't use vmap as arrays in arrs1d could have different shapes...
    return [_get_coordinate(a, v) for a, v in zip(arrs1d, values, strict=True)]


def _extract_strings(
    params: dict[str, Array | NDArray],
) -> tuple[dict[str, Array], dict[str, dict[str, int]]]:
    string_map = {}  # jax does not like strings
    new_params = {}
    for k, v in list(params.items()):
        if v.dtype == object:  # probably string array!
            string_map[k] = {s: i for i, s in enumerate(v)}
            new_params[k] = np.array(list(string_map[k].values()), dtype=int)
        else:
            new_params[k] = np.array(v, dtype=float)
    return new_params, string_map
