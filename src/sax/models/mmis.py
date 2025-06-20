"""SAX MMI Models."""

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax

from .couplers import coupler_ideal
from .splitters import splitter_ideal


@jax.jit
@validate_call
def mmi1x2_ideal(*, coupling: sax.FloatArrayLike = 0.5) -> sax.SDict:
    """A simple coupler model."""
    return splitter_ideal(coupling=coupling)


@jax.jit
@validate_call
def mmi2x2_ideal(*, coupling: sax.FloatArrayLike = 0.5) -> sax.SDict:
    """A simple coupler model."""
    return coupler_ideal(coupling=coupling)


@jax.jit
@validate_call
def mmi1x2(
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    fwhm: sax.FloatArrayLike = 0.2,
    loss_dB: sax.FloatArrayLike = 0.3,
) -> sax.SDict:
    """1x2 MMI model.

    Args:
        wl: wavelength.
        wl0: center wavelength.
        fwhm: full width at half maximum 3dB.
        loss_dB: loss in dB.
    """
    thru = _mmi_amp(wl=wl, wl0=wl0, fwhm=fwhm, loss_dB=loss_dB) / 2**0.5

    return sax.reciprocal(
        {
            ("o1", "o2"): thru,
            ("o1", "o3"): thru,
        }
    )


@jax.jit
@validate_call
def mmi2x2(
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    fwhm: sax.FloatArrayLike = 0.2,
    loss_dB: sax.FloatArrayLike = 0.3,
    shift: sax.FloatArrayLike = 0.005,
    loss_dB_cross: sax.FloatArrayLike | None = None,
    loss_dB_thru: sax.FloatArrayLike | None = None,
    splitting_ratio_cross: sax.FloatArrayLike = 0.5,
    splitting_ratio_thru: sax.FloatArrayLike = 0.5,
) -> sax.SDict:
    """2x2 MMI model.

    Args:
        wl: wavelength.
        wl0: center wavelength.
        fwhm: full width at half maximum.
        loss_dB: loss in dB.
        shift: shift in wavelength for both cross and thru ports.
        loss_dB_cross: loss in dB for the cross port.
        loss_dB_thru: loss in dB for the bar port.
        splitting_ratio_cross: splitting ratio for the cross port.
        splitting_ratio_thru: splitting ratio for the bar port.
    """
    loss_dB_cross = loss_dB_cross or loss_dB
    loss_dB_thru = loss_dB_thru or loss_dB

    # Convert splitting ratios from power to amplitude by taking the square root
    amplitude_ratio_thru = splitting_ratio_thru**0.5
    amplitude_ratio_cross = splitting_ratio_cross**0.5

    loss_factor_thru = 10 ** (-loss_dB_thru / 20)
    loss_factor_cross = 10 ** (-loss_dB_cross / 20)

    thru = (
        _mmi_amp(wl=wl, wl0=wl0 + shift, fwhm=fwhm, loss_dB=loss_dB_thru)
        * amplitude_ratio_thru
        * loss_factor_thru
    )
    cross = (
        1j
        * _mmi_amp(wl=wl, wl0=wl0 + shift, fwhm=fwhm, loss_dB=loss_dB_cross)
        * amplitude_ratio_cross
        * loss_factor_cross
    )

    return sax.reciprocal(
        {
            ("o1", "o3"): thru,
            ("o1", "o4"): cross,
            ("o2", "o3"): cross,
            ("o2", "o4"): thru,
        }
    )


def _mmi_amp(
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    fwhm: sax.FloatArrayLike = 0.2,
    loss_dB: sax.FloatArrayLike = 0.3,
) -> sax.FloatArray:
    max_power = 10 ** (-abs(loss_dB) / 10)
    f = 1 / wl
    f0 = 1 / wl0
    f1 = 1 / (wl0 + fwhm / 2)
    f2 = 1 / (wl0 - fwhm / 2)
    _fwhm = f2 - f1

    sigma = _fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    power = jnp.exp(-((f - f0) ** 2) / (2 * sigma**2))
    power = max_power * power / power.max()
    return jnp.sqrt(power)


def _mmi_nxn(
    n: int,
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    fwhm: sax.FloatArrayLike = 0.2,
    loss_dB: sax.FloatArrayLike | None = None,
    shift: sax.FloatArrayLike | None = None,
    splitting_matrix: sax.FloatArray2D | None = None,
) -> sax.SDict:
    """General n x n MMI model.

    Args:
        n (int): Number of input and output ports.
        wl (float): Operating wavelength.
        wl0 (float): Center wavelength of the MMI.
        fwhm (float): Full width at half maximum.
        loss_dB (np.array): Array of loss values in dB for each port.
        shift (np.array): Array of wavelength shifts for each port.
        splitting_matrix (np.array): nxn matrix defining the power splitting
            ratios between ports.
    """
    _loss_dB = jnp.zeros(n) if loss_dB is None else jnp.asarray(loss_dB)
    _shift = jnp.zeros(n) if shift is None else jnp.asarray(shift)
    _splitting_matrix = (
        jnp.full((n, n), 1 / n)
        if splitting_matrix is None
        else jnp.asarray(splitting_matrix)
    )

    S = {}
    for i in range(n):
        for j in range(n):
            amplitude = _mmi_amp(wl, wl0 + _shift[j], fwhm, _loss_dB[j])
            amplitude *= jnp.sqrt(_splitting_matrix[i][j])
            loss_factor = 10 ** (-_loss_dB[j] / 20)
            S[(f"o{i + 1}", f"o{j + 1}")] = amplitude * loss_factor

    return sax.reciprocal(S)
