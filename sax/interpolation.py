"""Simulation utilitites."""

from collections.abc import Sequence
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from jaxtyping import Array
from numpy.typing import NDArray
from scipy.constants import c


def interpolate_xarray(
    xarr: xr.DataArray,
    /,
    *,
    f: Array,
    **kwargs: Array,
) -> dict[str, Array]:
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

        params["targets"] = np.arange(0, len(targets), 1, dtype=np.uint8)  # type: ignore
        strings = {**strings, "targets": targets}

    S, axs, pos = evaluate_general_corner_model(
        data,
        params,  # type: ignore[reportArgumentType]
        strings,
        **kwargs,
        f=f,
    )
    return {k: S.take(pos["targets"][k], axs["targets"]) for k in targets}


def evaluate_general_corner_model(
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


def to_hypercube(
    stacked_data: pd.DataFrame, wl_key: str = "wl"
) -> tuple[Array, dict[str, Array], dict[str, dict[str, int]]]:
    """Converts a stacked dataframe into 'hypercube' representation.

    Args:
        stacked_data: the data to convert
        wl_key: the main wavelength key.

    Returns:
        data: the data to interpolate (output of to_hypercube)
        params: the parameter sample points (output of to_hypercube)
        strings: non-interpolatable parameters (like strings)
            with their values (output of to_hypercube)

    Returns:
        the interpolated S-matrix at the values of **kwargs
    """
    if wl_key not in ["f", "wl"]:
        msg = f"Unsupported wl_key. Valid choices: 'wl', 'f'. Got: {wl_key}."
        raise ValueError(msg)
    df = stacked_data.copy()
    df["f"] = c / df["wl"].to_numpy()
    value_columns = [c for c in ["amp", "phi"] if c in df.columns]
    param_columns = [c for c in df.columns if c not in [*value_columns, "wl", "f"]]
    param_columns = [c for c in [*param_columns, wl_key] if c in df.columns]
    value_columns = ["amp", "phi"] if len(value_columns) > 2 else value_columns
    df = _sort_rows(
        cast(pd.DataFrame, df[[*param_columns, *value_columns]]),
        not_by=tuple(value_columns),
        wl_key=wl_key,
    )
    params: dict[str, Array | NDArray] = {
        c: np.asarray(df[c].unique()) for c in param_columns
    }
    data = df[value_columns].to_numpy()
    data = data.reshape(*(v.shape[0] for v in params.values()), len(value_columns))
    data = jnp.asarray(data)
    params["targets"] = np.array(value_columns, dtype=object)
    float_params, string_params = _extract_strings(params)
    return data, float_params, string_params


def _sort_rows(
    df: pd.DataFrame, not_by: tuple[str, ...] = ("amp", "phi"), wl_key: str = "wl"
) -> pd.DataFrame:
    by = [c for c in df.columns if c not in ["wl", "f", *not_by]] + [wl_key]
    return df.sort_values(by=by).reset_index(drop=True)


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
