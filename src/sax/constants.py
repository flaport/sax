"""Constants and magic numbers."""

from __future__ import annotations

import jax.numpy as jnp

from sax.saxtypes.core import FloatArray1D

__all__ = [
    "C_M_S",
    "C_UM_S",
    "DEFAULT_MODE",
    "DEFAULT_MODES",
    "DEFAULT_WL_STEP",
    "EPS",
    "WL_C",
    "WL_C_MAX",
    "WL_C_MIN",
    "WL_E",
    "WL_E_MAX",
    "WL_E_MIN",
    "WL_L",
    "WL_L_MAX",
    "WL_L_MIN",
    "WL_O",
    "WL_O_MAX",
    "WL_O_MIN",
    "WL_S",
    "WL_S_MAX",
    "WL_S_MIN",
    "wl_c",
    "wl_e",
    "wl_l",
    "wl_o",
    "wl_s",
]

EPS: float = 1e-12
C_M_S: float = 299792458.0
C_UM_S: float = 1e6 * C_M_S
DEFAULT_MODE: str = "TE"
DEFAULT_MODES: tuple[str, ...] = ("TE", "TM")

DEFAULT_WL_STEP: float = 0.0001

# https://www.fiberlabs.com/glossary/optical-communication-band/
WL_O_MIN: float = 1.260
WL_O_MAX: float = 1.360
WL_O: float = 0.5 * (WL_O_MIN + WL_O_MAX)

WL_E_MIN: float = 1.360
WL_E_MAX: float = 1.460
WL_E: float = 0.5 * (WL_E_MIN + WL_E_MAX)

WL_S_MIN: float = 1.460
WL_S_MAX: float = 1.530
WL_S: float = 0.5 * (WL_S_MIN + WL_S_MAX)

WL_C_MIN: float = 1.530
WL_C_MAX: float = 1.565
WL_C: float = 0.5 * (WL_C_MIN + WL_C_MAX)

WL_L_MIN: float = 1.565
WL_L_MAX: float = 1.625
WL_L: float = 0.5 * (WL_L_MIN + WL_L_MAX)


def wl_o(
    *,
    step: float = DEFAULT_WL_STEP,
    num: int | None = None,
    wl_min: float = WL_O_MIN,
    wl_max: float = WL_O_MAX,
) -> FloatArray1D:
    """Wavelengths in the O-band."""
    return _wl(step=step, num=num, wl_min=wl_min, wl_max=wl_max)


def wl_e(
    *,
    step: float = DEFAULT_WL_STEP,
    num: int | None = None,
    wl_min: float = WL_E_MIN,
    wl_max: float = WL_E_MAX,
) -> FloatArray1D:
    """Wavelengths in the E-band."""
    return _wl(step=step, num=num, wl_min=wl_min, wl_max=wl_max)


def wl_s(
    *,
    step: float = DEFAULT_WL_STEP,
    num: int | None = None,
    wl_min: float = WL_S_MIN,
    wl_max: float = WL_S_MAX,
) -> FloatArray1D:
    """Wavelengths in the S-band."""
    return _wl(step=step, num=num, wl_min=wl_min, wl_max=wl_max)


def wl_c(
    *,
    step: float = DEFAULT_WL_STEP,
    num: int | None = None,
    wl_min: float = WL_C_MIN,
    wl_max: float = WL_C_MAX,
) -> FloatArray1D:
    """Wavelengths in the C-band."""
    return _wl(step=step, num=num, wl_min=wl_min, wl_max=wl_max)


def wl_l(
    *,
    step: float = DEFAULT_WL_STEP,
    num: int | None = None,
    wl_min: float = WL_L_MIN,
    wl_max: float = WL_L_MAX,
) -> FloatArray1D:
    """Wavelengths in the L-band."""
    return _wl(step=step, num=num, wl_min=wl_min, wl_max=wl_max)


def _wl(
    *,
    step: float,
    num: int | None,
    wl_min: float,
    wl_max: float,
) -> FloatArray1D:
    if num is not None:
        return jnp.linspace(wl_min, wl_max, num)
    return jnp.arange(wl_min, wl_max + step, step)
