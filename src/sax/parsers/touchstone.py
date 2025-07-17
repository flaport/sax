"""A touchstone data file parser."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import md5
from pathlib import Path
from tempfile import gettempdir

import numpy as np
import pandas as pd
import skrf as rf
import xarray as xr

import sax

__all__ = ["parse_touchstone"]


def parse_touchstone(
    content_or_filename: str | Path,
    *,
    ports: Iterable[str] = (),
    convert_to_wavelength: bool = True,
) -> pd.DataFrame:
    """Load S-parameters from a Lumerical .dat or .sparam file.

    Args:
        content_or_filename: Content as string (if contains newlines), file path,
            or file-like object with read() method.
        ports: port (or port@mode) labels to use.
            if not given, ports will be labeled as 'o1', 'o2', ...
        convert_to_wavelength: if True, convert frequency to wavelength.

    Returns:
        a pandas DataFrame with the S-parameters.

    Note:
        This function was adapted from the `simphony` load_sparams function.
    """
    temppath = None
    if isinstance(content_or_filename, str) and "\n" in content_or_filename:
        temppath = path = (
            Path(gettempdir()).resolve()
            / "sax"
            / f"touchstone_{md5(content_or_filename.encode()).hexdigest()}.dat"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content_or_filename)
    else:
        path = Path(content_or_filename).resolve()

    if not path.exists():
        msg = f"Touchstone file {path!r} not found."
        raise FileNotFoundError(msg)

    try:
        ntwk = rf.Network(str(path))
        ports = np.array(list(ports))
        if ports.shape[0] != ntwk.s.shape[-1] or ports.shape[0] != ntwk.s.shape[-2]:
            msg = (
                f"Length of ports list {ports} does not match "
                f"smatrix dimension [{ntwk.s.shape[-2]}x{ntwk.s.shape[-1]}]."
            )
            raise ValueError(msg)
        order = slice(None, None, -1) if convert_to_wavelength else slice(None)
        coords = {}
        if convert_to_wavelength:
            coords["wl"] = sax.C_UM_S / ntwk.f[order]
        coords["port_in"] = ports
        coords["port_out"] = ports
        xarr = xr.DataArray(ntwk.s[order], coords)
        df = sax.to_df(xarr, target_name="s")

        df["mode_in"] = [_get_mode(pm) for pm in df["port_in"].to_numpy()]
        df["mode_out"] = [_get_mode(pm) for pm in df["port_out"].to_numpy()]
        df["port_in"] = [_get_port(pm) for pm in df["port_in"].to_numpy()]
        df["port_out"] = [_get_port(pm) for pm in df["port_out"].to_numpy()]
        df["amp"] = np.abs(df["s"].to_numpy())
        df["phi"] = np.angle(df.pop("s").to_numpy())
    finally:
        if temppath is not None:
            temppath.unlink(missing_ok=True)
    return df


def _get_port(pm: str) -> str:
    return pm.split("@")[0]


def _get_mode(pm: str) -> str:
    _, *m = pm.split("@")
    return "".join(m) or "1"
