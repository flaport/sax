""" Load data files in OptSim format as SAX model functions

Note:
    This module is intended to load elementary model *functions* from data
    files in OptSim format. Loading full OptSim entities (for example from
    .moml files) is not supported (yet).

The OptSim data format has the following two-line header::

    TransferMatrixFormat5 <SubFormat>
    <num_inputs> <num_outputs> <num_wl> <min_wl> <max_wl>

Older TransferMatrixFormats (like ``TransferMatrixFormat4`` and below) are not
supported by SAX.

``<SubFormat>`` should be one of the following: ``POWER``, ``REAL``, ``REAL_IMAG``, ``AMP_PHASE``,
``AMP_PHASE_RAD``, ``POWER_PHASE``. (Other SubFormats are note supported by SAX)

The second line of the header can be extended with the following configurations:

  - ``<wavelength_units>``: either ``'wavelength_meters'`` or
    ``'wavelength_microns'`` (Other units are not supported by SAX)
  - ``polarization_mode=[0,1,2]``: ``0``: no polarization dependence, ``1``:
    no cross-polarization, ``2``: also cross-polarization
  - ``wavelength_grouping=[0,1,2]``: ``0``, ``1``: no wavelength grouping,
    ``2``: wavelength grouping for easier phase unwrapping
  - ``unwrap_phase=[0,1,2]``: ``0``: no unwrapping (assume unwrapped), ``1``:
    unwrap during interpolation, ``2``: unwrap while reading the file
    (default). (Not supported by SAX)
  - ``frequency_interp=[0,1]``: ``0``: interpolate on wavelength grid, ``1``:
    interpolate on frequency grid (default). (Not supported by SAX)

"""

from __future__ import annotations

import os
import sys
import warnings
import functools

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from typing import Dict, List, Tuple, NamedTuple, Callable
from .._typing import Array
from ..utils import zero

__all__ = [
    "load_optsim_df",
    "optsim_model_function",
    "amplitude_interpolation_with_grouping",
    "phase_interpolation_with_grouping",
]


def load_optsim_df(
    path: str,
    *s_params: str,
    from_cache: bool = True,
    save_cache: bool = True,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, _df_meta]:
    """Load (or partly load) an data file in OptSim format.

    Args:
        path: the path to the data file
        *s_params: the s parameters to load from the data file in OptSim
            format. Each s-parameter should be a string of the following format:
            "{input_port_idx}_{output_port_idx}_{input_mode_str}_{output_mode_str}"
            usually output_port_idx > input_port_idx and input_mode_str in ("te", "tm").
            If not given, all S-matrix elements will be loaded.
        from_cache: if possible, load the requested DataFrame from cache
        save_cache: pickle the resulting DataFrame for easier loading later.
        verbose: print info while loading the data

    Returns:
        the dataframe containing the data for the given s_params and a tuple
        containing some metadata about the data that was loaded.
    """

    path = os.path.abspath(os.path.expanduser(path))
    component, _ = os.path.splitext(os.path.basename(path))

    if s_params:
        _s_params = [str(param) for param in s_params]
        for i, _param in enumerate(_s_params):
            param = [el.lower().strip() for el in _param.split("_") if el]
            if (
                len(param) not in (2, 4)
                or param[2] not in ("te", "tm")
                or param[3] not in ("te", "tm")
            ):
                raise ValueError(
                    "Wrong format for s_params. Given s_params should be strings of "
                    "the following format: '{i_in}_{i_out}_{mode_in}_{mode_out}', "
                    "where mode_in and mode_out should be either 'TE' or 'TM'"
                )
            try:
                param[0] = str(int(param[0]))
                param[1] = str(int(param[1]))
            except ValueError:
                raise ValueError(
                    "Wrong format for s_params. Given s_params should be strings of "
                    "the following format: '{i_in}_{i_out}_{mode_in}_{mode_out}', "
                    "where i_in and i_out should be integer indices."
                )
            if len(param) == 2:
                param.extend(["te", "te"])
            _s_params[i] = "_".join(str(p) for p in param)
        s_params = tuple(_s_params)

    with open(path, "r") as file:
        path = os.path.abspath(os.path.expanduser(path))
        header1 = next(file).strip().split(" ")
        header2 = next(file).strip().split(" ")

    factorformat = ""
    if len(header1) == 2:
        transfer_matrix_format, subformat = header1
    elif len(header1) == 3:
        transfer_matrix_format, subformat, factorformat = header1
    else:
        raise ValueError(
            f"Error parsing '{path}'. " f"Invalid header (first line):\n\n{header1}"
        )

    if not transfer_matrix_format in ("TransferMatrixFormat5",):
        raise ValueError(
            f"Error parsing '{path}'. "
            f"Unsupported TransferMatrixFormat: '{transfer_matrix_format}'."
        )

    if not factorformat.upper() in ("",):
        raise ValueError(
            f"Error parsing '{path}'. " f"FactorFormat '{factorformat}' not supported."
        )

    _num_inputs, _num_outputs, _num_wl, _min_wl, _max_wl, *args_list = header2
    num_inputs = int(_num_inputs)
    num_outputs = int(_num_outputs)
    num_wl = int(_num_wl)
    min_wl = float(_min_wl)
    max_wl = float(_max_wl)

    possible_wavelength_units = (
        "wavelength_meters",
        "wavelength_microns",
        "frequency",
        "wavenumber",
    )
    wavelength_units_list = [
        arg for arg in args_list if arg in possible_wavelength_units
    ]
    if len(wavelength_units_list) > 1:
        raise ValueError(
            f"Error parsing '{path}': too many 'wavelength_units' "
            f"specified: {wavelength_units_list}"
        )
    wavelength_units = (
        "wavelength_meters" if not wavelength_units_list else wavelength_units_list[0]
    )
    if wavelength_units in ("frequency", "wavenumber"):
        raise ValueError(
            f"Error parsing '{path}': wavelength_units '{wavelength_units}' "
            f"not supported (yet)"
        )

    wavelength_factors = {"wavelength_meters": 1.0, "wavelength_microns": 1.0e-6}
    wavelength_factor = wavelength_factors[wavelength_units]

    names = []
    for i in range(num_inputs + num_outputs):
        if subformat in ("AMP_PHASE", "AMP_PHASE_RAD"):
            names.extend([f"A_n_{i}", f"P_n_{i}"])
        elif subformat in ("POWER_PHASE",):
            names.extend([f"A_n_{i}", f"P_n_{i}"])
        elif subformat in ("REAL_IMAG",):
            names.extend([f"R_n_{i}", f"I_n_{i}"])
        elif subformat in ("POWER", "REAL"):
            names.extend([f"R_n_{i}"])
        else:
            raise ValueError(
                f"Error parsing '{path}'. Subformat '{subformat}' not supported."
            )

    args = {
        k: int(v)
        for k, v in (
            arg.split("=") for arg in args_list if arg not in possible_wavelength_units
        )
    }

    # wavelength_grouping: 0, 1: no wavelength grouping, 2: wavelength grouping for easier phase unwrapping
    wavelength_grouping = args.get("wavelength_grouping", 0)
    if wavelength_grouping not in (0, 1, 2):
        raise ValueError(
            f"Error parsing '{path}': "
            f"wavelength_grouping={wavelength_grouping} not in [0, 1, 2]."
        )

    # unwrap_phase: 0: no unwrapping (assume unwrapped), 1: unwrap during interpolation, 2: unwrap while reading the file (default).
    unwrap_phase = args.get("unwrap_phase", 2)
    if unwrap_phase not in (2,):
        warnings.warn(
            f"unwrap_phase={unwrap_phase} not supported. "
            f"Please unwrap the phase manually."
        )

    # frequency_interp: 0: interpolate on wavelength grid, 1: interpolate on frequency grid (default).
    frequency_interp = args.get("frequency_interp", 1)
    if frequency_interp not in (1,):
        warnings.warn(
            f"frequency_interp={frequency_interp} not supported. "
            f"Please interpolate manually."
        )

    # polarization_mode: 0: no polarization dependendence, 1: no cross-polarization, 2: also cross-polarization
    polarization_mode = args.get("polarization_mode", 0)
    num_polarizations = 1
    dfs: Dict[str, List] = {}
    if polarization_mode in (0,):
        names = [f"{name}_te_te" for name in names]
        num_polarizations = 1
        dfs["te"] = []
    elif polarization_mode in (1,):
        names = [f"{name}_tem_tem" for name in names]
        num_polarizations = 2
        dfs["te"] = []
        dfs["tm"] = []
    elif polarization_mode in (2,):
        names_te = [f"{name}_tem_te" for name in names]
        names_tm = [f"{name}_tem_tm" for name in names]
        names = names_te + names_tm
        num_polarizations = 2
        dfs["te"] = []
        dfs["tm"] = []
    else:
        raise ValueError(
            f"Error parsing '{path}': "
            f"polarization_mode={polarization_mode} not in [0, 1, 2]."
        )

    meta = _df_meta(
        component,
        num_inputs,
        num_outputs,
        num_wl,
        min_wl,
        max_wl,
        wavelength_grouping,
        unwrap_phase,
        frequency_interp,
        polarization_mode,
    )
    cached_path = os.path.join(
        os.path.dirname(path),
        "__pycache__",
        "-".join(str(x) for x in meta)
        + ("-ALL" if not s_params else "-")
        + "-".join(str(x) for x in s_params)
        + ".pkl",
    )

    if from_cache and os.path.exists(cached_path):
        df = pd.read_pickle(cached_path).copy()
        if verbose:
            print(
                f"Loaded from cache. "
                f"Memory usage: {df.memory_usage().sum()/(1024**2):.0f}Mb",
                file=sys.stderr,
            )
        return df, meta

    def _handle_df_chunk(df, idx=0, subformat=subformat):
        if idx * num_wl < total_size // num_polarizations:
            input_mode = "te"
        else:
            input_mode = "tm"
            idx -= int(total_size) // int(num_polarizations) // int(num_wl)
        df["wl"] *= wavelength_factor
        df = df.set_index("wl")
        df.columns = [
            name.replace("_n_", f"_{idx}_").replace("_tem", f"_{input_mode}")
            for name in df.columns
        ]
        if s_params:
            df = df[
                [
                    name
                    for name in df.columns
                    if any(f"_{param}" in name for param in s_params)
                ]
            ]

        if subformat in ("POWER_PHASE",):
            c_a = [name for name in df.columns if name.startswith("A_")]
            df[c_a] = np.sqrt(df[c_a])
            subformat = "AMP_PHASE"

        if subformat in ("AMP_PHASE",):
            c_p = [name for name in df.columns if name.startswith("P_")]
            df[c_p] *= np.pi / 180.0
            subformat = "AMP_PHASE_RAD"

        if subformat in ("POWER",):
            c_r = [name for name in df.columns if name.startswith("R_")]
            df[c_r] = np.sqrt(df[c_r])
            subformat = "REAL"

        if subformat in ("REAL",):
            c_r = np.array(
                [name for name in df.columns if name.startswith("R_")], dtype=object
            )
            c_i = np.array(
                [
                    name.replace("R_", "I_")
                    for name in df.columns
                    if name.startswith("R_")
                ],
                dtype=object,
            )
            v_r = df[c_r].values
            v_i = np.zeros_like(v_r)
            v = np.stack([v_r, v_i], -1).reshape(v_r.shape[0], -1)
            c = np.stack([c_r, c_i], 1).ravel()
            wl = df.index
            df = pd.DataFrame(data=v, columns=c)
            df["wl"] = wl
            df = df.set_index("wl")
            subformat = "REAL_IMAG"

        if subformat in ("REAL_IMAG",):
            c_r = [name for name in np.array(df.columns) if name.startswith("R_")]
            c_i = [name for name in np.array(df.columns) if name.startswith("I_")]
            v_r = np.array(df[c_r].values)
            v_i = np.array(df[c_i].values)
            v_a = np.sqrt(v_r ** 2 + v_i ** 2)
            v_p = np.arctan2(v_i, v_r)
            v = np.stack([v_a, v_p], -1).reshape(v_a.shape[0], -1)
            c = [
                name.replace("R_", "A_").replace("I_", "P_")
                for name in np.array(df.columns)
            ]
            wl = df.index
            df = pd.DataFrame(data=v, columns=c)
            df["wl"] = wl
            df = df.set_index("wl")
            subformat = "AMP_PHASE_RAD"

        return df, input_mode

    num_rows = 0
    memory = 0.0
    total_size = int(num_wl * num_polarizations * (num_inputs + num_outputs))
    df_reader = pd.read_csv(
        path,
        skiprows=2,
        sep=" ",
        names=["wl"] + names,
        header=None,
        index_col=False,
        chunksize=num_wl,
    )
    for idx, df in enumerate(df_reader):
        df, input_mode = _handle_df_chunk(df, idx=idx, subformat=subformat)
        num_rows += df.shape[0]
        memory += df.memory_usage().sum() / (1024 ** 2)
        dfs[input_mode].append(df)
        if verbose:
            print(
                f"{num_rows} rows loaded. Memory usage: {memory:.0f}Mb",
                end="\r",
                file=sys.stderr,
            )

    if polarization_mode == 1:
        df = pd.concat(dfs["te"], 1)
    else:
        df = pd.concat([pd.concat(dfs["te"], 1), pd.concat(dfs["tm"], 1)], 1)

    if save_cache:
        os.makedirs(os.path.dirname(cached_path), exist_ok=True)
        df.to_pickle(cached_path)

    return df, meta


def optsim_model_function(
    path: str,
    input_port: int = 0,
    output_port: int = 1,
    input_mode: str = "te",
    output_mode: str = "te",
    from_cache: bool = True,
    save_cache: bool = True,
) -> Callable:
    """generate a model function from a datafile in OptSim format

    Args:
        path: location of the datafile in OptSim format.
        input_port: the index of the input port
        output_port: the index of the output port
        input_mode: the input mode ('TE' or 'TM')
        output_mode: the input mode ('TE' or 'TM')
        from_cache: if possible, load the requested data file from cache
        save_cache: pickle the resulting data for easier loading later.

    Returns:
        a model function which interpolated the given data file
    """
    path = os.path.abspath(os.path.expanduser(str(path)))
    input_port = int(input_port)
    output_port = int(output_port)
    input_mode = str(input_mode).lower()
    output_mode = str(output_mode).lower()
    assert input_mode in ("te", "tm"), "mode0 should be 'TE' or 'TM'"
    assert output_mode in ("te", "tm"), "mode1 should be 'TE' or 'TM'"
    s_param = f"{input_port}_{output_port}_{input_mode}_{output_mode}"
    df, meta = load_optsim_df(
        path,
        from_cache=from_cache,
        save_cache=save_cache
        # path, s_param, from_cache=from_cache, save_cache=save_cache
    )
    if df.values.size == 0:
        return zero
    wls = jnp.array(df.index.values)
    phi = jnp.array(df[f"P_{s_param}"].values.ravel())
    amp = jnp.array(df[f"A_{s_param}"].values.ravel())

    phase_interpolation = (
        phase_interpolation_with_grouping
        if meta.wavelength_grouping == 2
        else jnp.interp
    )

    def optsim_model_func(params):
        assert isinstance(wls, jnp.ndarray)
        assert isinstance(phi, jnp.ndarray)
        wl = jnp.atleast_1d(params["wl"])
        assert isinstance(wl, jnp.ndarray)
        phase = phase_interpolation(wl, wls, phi)
        assert isinstance(phase, jnp.ndarray)
        amplitude = jnp.interp(wl, wls, amp)
        return amplitude * jnp.exp(1j * phase)

    assert isinstance(wls, jnp.ndarray)
    optsim_model_func.__doc__ = f""" SAX model function for {meta.component}

    Connection:
        {input_port}[{input_mode}] -> {output_port}[{output_mode}]

    Args:
        params: parameter dictionary containing at least the key 'wl' for wavelength.

    Note:
        this interpolation is only accurate in the wavelength
        range [{wls[0]*1e6:.3f}μm, {wls[-2]*1e6:.3f}μm) (up to but not including the last wavelength).
        Any extrapolation outside these bounds can yield unexpected results!

    Data:
        the data for this model function can be found here:
        '{path}'
    """

    return optsim_model_func


def _vmap_interpolation(func):
    return functools.wraps(func)(jax.vmap(func, in_axes=(0, None, None), out_axes=0))


@_vmap_interpolation
def amplitude_interpolation_with_grouping(wl: Array, wls: Array, amp: Array) -> Array:
    """Interpolate amplitude where the given wavelengths and amplitudes have wavelength grouping

    Args:
        wl: the wavelength points at which to evaluate the interpolation
        wls: given wavelengths with known amplitude values. Every two
            wavelengths should be paired close together (wavelength grouping
            format) as to enable accurate amplitude interpolation.
        amp: given amplitudes corresponding to the given wavelengths

    Returns:
        amplitude values for the wavelengths to interpolate for

    Note:
        this interpolation is only accurate in the
        range [wls[0], wls[-2]) (wls[-2] not included).
        Any extrapolation outside these bounds can yield unexpected results!
    """
    damp_dwl = (amp[1::2] - amp[::2]) / (wls[1::2] - wls[::2])
    amp = amp[::2]
    wls = wls[::2]
    dwl = (wls[1:] - wls[:-1]).mean(0, keepdims=True)
    t = (jax.lax.nextafter(wl, wl + dwl) - wls) / dwl
    t = jnp.where(jnp.abs(t) < 1, t, 0)
    assert isinstance(t, jnp.ndarray)
    m0 = jnp.where(t > 0, 1.0, 0.0)
    assert isinstance(m0, jnp.ndarray)
    m1 = jnp.where(t < 0, 1.0, 0.0)
    assert isinstance(m1, jnp.ndarray)
    t = (t * m0).sum(0)
    wl0 = (wls * m0).sum(0)
    wl1 = (wls * m1).sum(0)
    amp0 = (amp * m0).sum(0)
    amp1 = (amp * m1).sum(0)
    damp_dwl0 = (damp_dwl * m0).sum(0)
    damp_dwl1 = (damp_dwl * m1).sum(0)
    _amp0 = amp0 - 0.5 * (wl1 - wl0) * (
        damp_dwl0 * (t ** 2 - 2 * t) - damp_dwl1 * t ** 2
    )
    _amp1 = amp1 - 0.5 * (wl1 - wl0) * (
        damp_dwl0 * (t - 1) ** 2 - damp_dwl1 * (t ** 2 - 1)
    )
    _amp = (1 - t) * _amp0 + t * _amp1
    return _amp


@_vmap_interpolation
def phase_interpolation_with_grouping(wl: Array, wls: Array, phi: Array) -> Array:
    """Interpolate phase where the given wavelengths and phases have wavelength grouping

    Args:
        wl: the wavelength points at which to evaluate the interpolation
        wls: given wavelengths with known phase values. Every two wavelengths
            should be paired close together (wavelength grouping format) as to
            enable accurate phase interpolation.
        phis: given phases corresponding to the given wavelengths

    Returns:
        phase values for the wavelengths to interpolate for

    Note:
        this interpolation is only accurate in the
        range [wls[0], wls[-2]) (wls[-2] not included).
        Any extrapolation outside these bounds can yield unexpected results!
    """
    dphi_dwl = (phi[1::2] - phi[::2]) / (wls[1::2] - wls[::2])
    phi = phi[::2]
    wls = wls[::2]
    dwl = (wls[1:] - wls[:-1]).mean(0, keepdims=True)
    t = (wl - wls + dwl * 1e-5) / dwl
    t = jnp.where(jnp.abs(t) < 1, t, 0)
    assert isinstance(t, jnp.ndarray)
    m0 = jnp.where(t > 0, 1.0, 0.0)
    assert isinstance(m0, jnp.ndarray)
    m1 = jnp.where(t < 0, 1.0, 0.0)
    assert isinstance(m1, jnp.ndarray)
    t = (t * m0).sum(0)
    wl0 = (wls * m0).sum(0)
    wl1 = (wls * m1).sum(0)
    phi0 = (phi * m0).sum(0)
    phi1 = (phi * m1).sum(0)
    dphi_dwl0 = (dphi_dwl * m0).sum(0)
    dphi_dwl1 = (dphi_dwl * m1).sum(0)
    _phi0 = phi0 - 0.5 * (wl1 - wl0) * (
        dphi_dwl0 * (t ** 2 - 2 * t) - dphi_dwl1 * t ** 2
    )
    _phi1 = phi1 - 0.5 * (wl1 - wl0) * (
        dphi_dwl0 * (t - 1) ** 2 - dphi_dwl1 * (t ** 2 - 1)
    )
    phi = jnp.arctan2(
        (1 - t) * jnp.sin(_phi0) + t * jnp.sin(_phi1),
        (1 - t) * jnp.cos(_phi0) + t * jnp.cos(_phi1),
    )
    return phi


class _df_meta(NamedTuple):
    """ OptSim Dataframe metadata """

    component: str
    num_inputs: int
    num_outputs: int
    num_wl: int
    min_wl: float
    max_wl: float
    wavelength_grouping: int
    unwrap_phase: int
    frequency_interp: int
    polarization_mode: int
