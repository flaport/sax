"""Constants and magic numbers."""

import jax.numpy as jnp

from sax.saxtypes.core import FloatArray1D

EPS: float = 1e-12
C_M_S: float = 299792458.0
C_UM_S: float = 1e6 * C_M_S
WL_O_MIN: float = 1.260
WL_O_MAX: float = 1.360
WL_O: float = 0.5 * (WL_O_MIN + WL_O_MAX)
WL_C_MIN: float = 1.530
WL_C_MAX: float = 1.565
WL_C: float = 0.5 * (WL_C_MIN + WL_C_MAX)
DEFAULT_WL_STEP: float = 0.001
DEFAULT_MODE: str = "TE"
DEFAULT_MODES: tuple[str, ...] = ("TE", "TM")
WLS_O: FloatArray1D = jnp.arange(WL_O_MIN, WL_O_MAX + DEFAULT_WL_STEP, DEFAULT_WL_STEP)
WLS_C: FloatArray1D = jnp.arange(WL_C_MIN, WL_C_MAX + DEFAULT_WL_STEP, DEFAULT_WL_STEP)
