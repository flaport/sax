"""SAX Coupler Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def coupler_ideal(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    coupling: sax.FloatArrayLike = 0.5,
) -> sax.SDict:
    r"""Ideal 2x2 directional coupler model.

    ```{svgbob}
     in1          out1
      o2          o3
       *          *
        \        /
         '------'
         coupling
         .------.
        /        \
       *          *
      o1          o4
     in0          out0
    ```

    Args:
        wl: the wavelength of the simulation in micrometers.
        coupling: Power coupling coefficient between 0 and 1. Defaults to 0.5.

    Returns:
        The coupler s-matrix

    Examples:
        Ideal coupler:

        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        wl = sax.wl_c()
        s = sax.models.coupler_ideal(
            wl=wl,
            coupling=0.3,
        )
        thru = np.abs(s[("o1", "o4")]) ** 2
        cross = np.abs(s[("o1", "o3")]) ** 2
        plt.figure()
        plt.plot(wl, thru, label="thru")
        plt.plot(wl, cross, label="cross")
        plt.xlabel("Wavelength [μm]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    one = jnp.ones_like(wl)
    kappa = jnp.asarray(coupling**0.5) * one
    tau = jnp.asarray((1 - coupling) ** 0.5) * one
    p = sax.PortNamer(2, 2)
    return sax.reciprocal(
        {
            (p.in0, p.out0): tau,
            (p.in0, p.out1): 1j * kappa,
            (p.in1, p.out0): 1j * kappa,
            (p.in1, p.out1): tau,
        },
    )


@jax.jit
@validate_call
def coupler(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    wl0: sax.FloatArrayLike = sax.WL_C,
    length: sax.FloatArrayLike = 0.0,
    coupling0: sax.FloatArrayLike = 0.2,
    dk1: sax.FloatArrayLike = 1.2435,
    dk2: sax.FloatArrayLike = 5.3022,
    dn: sax.FloatArrayLike = 0.02,
    dn1: sax.FloatArrayLike = 0.1169,
    dn2: sax.FloatArrayLike = 0.4821,
) -> sax.SDict:
    r"""Dispersive directional coupler model.

    ```{svgbob}
            in1                out1
             o2                o3
              *                *
               \              /
                '------------'
    coupling0 /2   coupling   coupling0 /2
                .------------.
               /  <-length -> \
              *                *
             o1                o4
            in0                out0
    ```

    Args:
        wl: Operating wavelength in micrometers. Can be a scalar or array for
            multi-wavelength simulations. Defaults to 1.55 μm.
        wl0: Reference wavelength in micrometers for dispersion expansion.
            Typically the center design wavelength. Defaults to 1.55 μm.
        length: Coupling length in micrometers. This is the length over which
            the two waveguides are in close proximity. Defaults to 0.0 μm.
        coupling0: Base coupling coefficient at the reference wavelength from
            bend regions or other coupling mechanisms. Obtained from FDTD
            simulations or measurements. Defaults to 0.2.
        dk1: First-order derivative of coupling coefficient with respect to
            wavelength (∂κ/∂λ). Units: μm⁻¹. Defaults to 1.2435.
        dk2: Second-order derivative of coupling coefficient with respect to
            wavelength (∂²κ/∂λ²). Units: μm⁻². Defaults to 5.3022.
        dn: Effective index difference between even and odd supermodes at
            the reference wavelength. Determines beating length. Defaults to 0.02.
        dn1: First-order derivative of effective index difference with respect
            to wavelength (∂Δn/∂λ). Units: μm⁻¹. Defaults to 0.1169.
        dn2: Second-order derivative of effective index difference with respect
            to wavelength (∂²Δn/∂λ²). Units: μm⁻². Defaults to 0.4821.

    Returns:
        The coupler s-matrix

    Examples:
        Basic dispersive coupler:

    ```python
    # mkdocs: render
    import matplotlib.pyplot as plt
    import numpy as np
    import sax

    sax.set_port_naming_strategy("optical")

    wl = sax.wl_c()
    s = sax.models.coupler(
        wl=wl,
        length=15.7,
        coupling0=0.0,
        dn=0.02,
    )
    thru = np.abs(s[("o1", "o4")]) ** 2
    cross = np.abs(s[("o1", "o3")]) ** 2
    plt.figure()
    plt.plot(wl, thru, label="thru")
    plt.plot(wl, cross, label="cross")
    plt.xlabel("Wavelength [μm]")
    plt.ylabel("Power")
    plt.legend()
    ```

    Note:
        The coupling strength follows the formula:

        κ_total = κ₀ + κ₁ + κ_length

        Where:

        - κ₀ = coupling0 + dk1*(λ-λ₀) + 0.5*dk2*(λ-λ₀)²
        - κ₁ = π*Δn(λ)/λ
        - κ_length represents the distributed coupling over the interaction length

    """
    dwl = wl - wl0
    dn = dn + dn1 * dwl + 0.5 * dn2 * dwl**2
    kappa0 = coupling0 + dk1 * dwl + 0.5 * dk2 * dwl**2
    kappa1 = jnp.pi * dn / wl

    tau = jnp.cos(kappa0 + kappa1 * length)
    kappa = -jnp.sin(kappa0 + kappa1 * length)
    p = sax.PortNamer(2, 2)
    return sax.reciprocal(
        {
            (p.in0, p.out0): tau,
            (p.in0, p.out1): 1j * kappa,
            (p.in1, p.out0): 1j * kappa,
            (p.in1, p.out1): tau,
        }
    )


@validate_call
def grating_coupler(
    *,
    wl: sax.FloatArrayLike = sax.WL_C,
    wl0: sax.FloatArrayLike = sax.WL_C,
    loss: sax.FloatArrayLike = 0.0,
    reflection: sax.FloatArrayLike = 0.0,
    reflection_fiber: sax.FloatArrayLike = 0.0,
    bandwidth: sax.FloatArrayLike = 40e-3,
) -> sax.SDict:
    """Grating coupler model for fiber-chip coupling (2-port reciprocal S-matrix).

    ```{svgbob}
                  out0
                  o2
             /  /  /  /
            /  /  /  /  fiber
           /  /  /  /
           _   _   _
     o1  _| |_| |_| |__
    in0
    ```

    Args:
        wl: Operating wavelength(s) in micrometers. Scalar or array.
        wl0: Center wavelength (μm) where peak transmission occurs.
        loss: Insertion loss (dB) at wl0, applied to *power*. Must be in [0, 20].
            Converted internally to an *amplitude* factor A0 = 10^(-loss/20).
        reflection: **Amplitude** reflection coefficient seen from the waveguide
            side (port in0). Must be in [0, 1]. (If you have reflection specified
            as *power*, convert via sqrt first.)
        reflection_fiber: **Amplitude** reflection coefficient seen from the
            fiber side (port out0). Must be in [0, 1].
        bandwidth: 3 dB bandwidth (FWHM) in micrometers (e.g., 40e-3 = 40 nm).

    Returns:
        The grating coupler s-matrix

    Raises:
        ValueError: If reflection or reflection_fiber is outside [0, 1],
            or if loss is outside [0, 20] dB.

    Examples:
        Basic grating coupler:

    ```python
    # mkdocs: render
    import matplotlib.pyplot as plt
    import numpy as np
    import sax

    sax.set_port_naming_strategy("optical")

    wl = np.linspace(1.5, 1.6, 101)
    s = sax.models.grating_coupler(
        wl=wl,
        loss=3.0,
        bandwidth=0.035,
    )
    plt.figure()
    plt.plot(wl, np.abs(s[("o1", "o2")]) ** 2, label="Transmission")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Power")
    plt.legend()
    ```

    Note:
        Amplitude transmission has a Gaussian spectral envelope:
            A(λ) = A0 * exp(-(λ-λ0)^2 / (4σ^2))
        with σ = FWHM / (2*sqrt(2*ln(2))).

        This model assumes:

        - Gaussian spectral response (typical for uniform gratings)
        - Wavelength-independent loss coefficient
        - Linear polarization (TE or TM)
        - Single-mode fiber coupling
        - No higher-order diffraction effects

    """
    import numpy as np

    # Validate reflection coefficients (amplitude, must be in [0, 1])
    r_arr = np.asarray(reflection)
    if np.any(r_arr < 0) or np.any(r_arr > 1):
        raise ValueError(
            f"reflection must be in [0, 1] (amplitude), got {reflection}. "
            "If you have power reflection, use sqrt(R_power) instead."
        )

    r_fib_arr = np.asarray(reflection_fiber)
    if np.any(r_fib_arr < 0) or np.any(r_fib_arr > 1):
        raise ValueError(
            f"reflection_fiber must be in [0, 1] (amplitude), got {reflection_fiber}. "
            "If you have power reflection, use sqrt(R_power) instead."
        )

    # Validate loss (dB, typically 0-20 dB for grating couplers)
    loss_arr = np.asarray(loss)
    if np.any(loss_arr < 0) or np.any(loss_arr > 20):
        raise ValueError(
            f"loss must be in [0, 20] dB, got {loss}. "
            "Note: loss is in dB (e.g., 3.0 for 3 dB loss), not linear."
        )

    return _grating_coupler_impl(
        wl=wl,
        wl0=wl0,
        loss=loss,
        reflection=reflection,
        reflection_fiber=reflection_fiber,
        bandwidth=bandwidth,
    )


@jax.jit
def _grating_coupler_impl(
    wl: sax.FloatArrayLike,
    wl0: sax.FloatArrayLike,
    loss: sax.FloatArrayLike,
    reflection: sax.FloatArrayLike,
    reflection_fiber: sax.FloatArrayLike,
    bandwidth: sax.FloatArrayLike,
) -> sax.SDict:
    """JIT-compiled implementation of grating_coupler."""
    wl = jnp.asarray(wl)
    wl0 = jnp.asarray(wl0)
    loss = jnp.asarray(loss)
    reflection = jnp.asarray(reflection)
    reflection_fiber = jnp.asarray(reflection_fiber)
    bandwidth = jnp.asarray(bandwidth)

    # Broadcast scalars/short arrays to wl shape
    wl_shape = wl.shape
    wl0_b = jnp.broadcast_to(wl0, wl_shape)
    loss_b = jnp.broadcast_to(loss, wl_shape)
    r_wg = jnp.broadcast_to(reflection, wl_shape)
    r_fib = jnp.broadcast_to(reflection_fiber, wl_shape)
    bw_b = jnp.broadcast_to(bandwidth, wl_shape)

    # Constants + conversions
    ln2 = jnp.log(jnp.asarray(2.0, dtype=wl.dtype))
    sigma = bw_b / (2.0 * jnp.sqrt(2.0 * ln2))  # FWHM -> σ
    a0 = 10.0 ** (-loss_b / 20.0)  # dB (power) -> amplitude

    # Gaussian amplitude envelope
    d = wl - wl0_b
    t = a0 * jnp.exp(-(d * d) / (4.0 * sigma * sigma))

    p = sax.PortNamer(1, 1)
    return sax.reciprocal(
        {
            (p.in0, p.in0): r_wg,
            (p.in0, p.out0): t,
            (p.out0, p.in0): t,
            (p.out0, p.out0): r_fib,
        }
    )
