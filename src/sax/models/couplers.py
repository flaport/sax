"""SAX Coupler Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def coupler_ideal(*, coupling: sax.FloatArrayLike = 0.5) -> sax.SDict:
    r"""Ideal 2x2 directional coupler model.

    This function models an ideal 2x2 directional coupler with perfect coupling
    efficiency and no loss. The coupler divides input power between two output
    ports based on the coupling coefficient, with a 90-degree phase shift
    between cross-coupled terms to maintain unitarity.

    The coupler has four ports arranged as:
    - Input ports: in0, in1 (left side)
    - Output ports: out0, out1 (right side)

    Power coupling behavior:
    - Fraction (1-coupling) transmits straight through (bar state)
    - Fraction coupling is cross-coupled between arms
    - Total power is conserved across all ports

    Args:
        coupling: Power coupling coefficient between 0 and 1. Determines the
            fraction of power that couples between the two waveguide arms.
            - coupling=0.0: No coupling (straight transmission)
            - coupling=0.5: 3dB coupler (equal bar/cross transmission)
            - coupling=1.0: Complete cross-coupling
            Defaults to 0.5.

    Returns:
        The coupler s-matrix

    Examples:
        3dB coupler (equal splitting):

        ```python
        import sax

        s_matrix = sax.models.coupler_ideal(coupling=0.5)
        # Bar transmission (straight through)
        bar_power = abs(s_matrix[("in0", "out0")]) ** 2
        # Cross transmission (between arms)
        cross_power = abs(s_matrix[("in0", "out1")]) ** 2
        print(f"Bar power: {bar_power:.3f}")  # Should be 0.5
        print(f"Cross power: {cross_power:.3f}")  # Should be 0.5
        ```

        Asymmetric coupler:

        ```python
        s_matrix = sax.models.coupler_ideal(coupling=0.1)  # 10% coupling
        bar_power = abs(s_matrix[("in0", "out0")]) ** 2  # Should be 0.9
        cross_power = abs(s_matrix[("in0", "out1")]) ** 2  # Should be 0.1
        ```

        Phase relationship verification:

        ```python
        s_matrix = sax.models.coupler_ideal(coupling=0.5)
        bar_phase = jnp.angle(s_matrix[("in0", "out0")])
        cross_phase = jnp.angle(s_matrix[("in0", "out1")])
        phase_diff = cross_phase - bar_phase
        print(f"Phase difference: {phase_diff:.3f} rad")  # Should be π/2
        ```

    Note:
        This is an idealized lossless model that assumes:
        - Perfect power conservation (unitary S-matrix)
        - No reflection at any port
        - Wavelength-independent behavior
        - Symmetric coupling between both directions

        The 90-degree phase shift in cross-coupled terms (1j factor) is required
        for maintaining S-matrix unitarity and represents the physical phase
        relationship in evanescent coupling.

    """
    kappa = jnp.asarray(coupling**0.5)
    tau = jnp.asarray((1 - coupling) ** 0.5)
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
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    length: sax.FloatArrayLike = 0.0,
    coupling0: sax.FloatArrayLike = 0.2,
    dk1: sax.FloatArrayLike = 1.2435,
    dk2: sax.FloatArrayLike = 5.3022,
    dn: sax.FloatArrayLike = 0.02,
    dn1: sax.FloatArrayLike = 0.1169,
    dn2: sax.FloatArrayLike = 0.4821,
) -> sax.SDict:
    r"""Dispersive directional coupler model.

    This function models a realistic directional coupler with wavelength-dependent
    coupling and chromatic dispersion effects. The model includes both the
    coupling region dispersion and bend-induced coupling contributions.

    The model is based on coupled-mode theory and includes:
    - Wavelength-dependent coupling coefficient
    - Effective index difference between even/odd supermodes
    - Both first and second-order dispersion terms
    - Bend region coupling contributions

    equations adapted from photontorch.
    https://github.com/flaport/photontorch/blob/master/photontorch/components/directionalcouplers.py

    kappa = coupling0 + coupling

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
        import sax
        import numpy as np

        s_matrix = sax.models.coupler(
            wl=1.55,
            length=10.0,  # 10 μm coupling length
            coupling0=0.1,
            dn=0.015
        )
        bar_transmission = abs(s_matrix[('in0', 'out0')])**2
        cross_transmission = abs(s_matrix[('in0', 'out1')])**2
        ```

        Wavelength sweep analysis:

        ```python
        wavelengths = np.linspace(1.5, 1.6, 101)
        s_matrices = sax.models.coupler(
            wl=wavelengths,
            length=20.0,
            coupling0=0.2,
            dn=0.02,
            dn1=0.1  # Include dispersion
        )
        bar_power = np.abs(s_matrices[('in0', 'out0')])**2
        cross_power = np.abs(s_matrices[('in0', 'out1')])**2
        ```

        Design for specific coupling:

        ```python
        # Design for 3dB coupling at 1.55 μm
        target_coupling = 0.5
        # Adjust length and coupling0 to achieve target
        s_matrix = sax.models.coupler(
            wl=1.55,
            length=15.7,  # Calculated for π/2 phase
            coupling0=0.0,
            dn=0.02
        )
        ```

    ```
        in1/o2 -----                      ----- out1/o3
                    \ ◀-----length-----▶ /
                    --------------------
        coupling0/2      coupling      coupling0/2
                    --------------------
                    /                    \
        in0/o1 -----                      ----- out0/o4

    ```

    ```python
    # mkdocs: render
    import matplotlib.pyplot as plt
    import numpy as np
    import sax

    sax.set_port_naming_strategy("optical")

    wavelengths = np.linspace(1.5, 1.6, 101)
    s = sax.models.coupler(
        wl=wavelengths,
        length=15.7,
        coupling0=0.0,
        dn=0.02,
    )
    bar_power = np.abs(s[("o1", "o4")]) ** 2
    cross_power = np.abs(s[("o1", "o3")]) ** 2
    plt.figure()
    plt.plot(wavelengths, bar_power, label="Bar")
    plt.plot(wavelengths, cross_power, label="Cross")
    plt.xlabel("Wavelength (μm)")
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

        This model assumes:
        - Weak coupling regime (small coupling per unit length)
        - Linear dispersion approximation for small wavelength deviations
        - Symmetric coupler geometry
        - No higher-order modes
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


@jax.jit
@validate_call
def grating_coupler(
    *,
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    loss: sax.FloatArrayLike = 0.0,
    reflection: sax.FloatArrayLike = 0.0,
    reflection_fiber: sax.FloatArrayLike = 0.0,
    bandwidth: sax.FloatArrayLike = 40e-3,
) -> sax.SDict:
    """Grating coupler model for fiber-chip coupling.

    This function models a grating coupler used to couple light between an
    optical fiber and an on-chip waveguide. The model includes wavelength-
    dependent transmission with a Gaussian spectral response, insertion loss,
    and reflection effects from both the waveguide and fiber sides.

    equation adapted from photontorch grating coupler
    https://github.com/flaport/photontorch/blob/master/photontorch/components/gratingcouplers.py

    The grating coupler provides:
    - Vertical coupling between fiber and chip
    - Wavelength-selective transmission
    - Bidirectional operation
    - Reflection handling for both interfaces

    Args:
        wl: Operating wavelength in micrometers. Can be a scalar or array for
            spectral analysis. Defaults to 1.55 μm.
        wl0: Center wavelength in micrometers where peak transmission occurs.
            This is the design wavelength of the grating. Defaults to 1.55 μm.
        loss: Insertion loss in dB at the center wavelength. Includes coupling
            efficiency losses, scattering, and mode mismatch. Defaults to 0.0 dB.
        reflection: Reflection coefficient from the waveguide side (chip interface).
            Represents reflections back into the waveguide from grating discontinuities.
            Range: 0 to 1. Defaults to 0.0.
        reflection_fiber: Reflection coefficient from the fiber side (top interface).
            Represents reflections back toward the fiber from the grating surface.
            Range: 0 to 1. Defaults to 0.0.
        bandwidth: 3dB bandwidth in micrometers. Determines the spectral width
            of the Gaussian transmission profile. Typical values: 20-50 nm.
            Defaults to 40e-3 μm (40 nm).

    Returns:
        The grating coupler s-matrix

    Examples:
        Basic grating coupler:

        ```python
        import sax
        import numpy as np

        s_matrix = sax.models.grating_coupler(
            wl=1.55,
            wl0=1.55,
            loss=3.0,  # 3 dB insertion loss
            bandwidth=0.035,  # 35 nm bandwidth
        )
        coupling_efficiency = abs(s_matrix[("in0", "out0")]) ** 2
        print(f"Coupling efficiency: {coupling_efficiency:.3f}")
        ```

        Spectral response analysis:

        ```python
        wavelengths = np.linspace(1.5, 1.6, 101)
        s_matrices = sax.models.grating_coupler(
            wl=wavelengths, wl0=1.55, loss=4.0, bandwidth=0.040
        )
        transmission = np.abs(s_matrices[("in0", "out0")]) ** 2
        # Plot spectral response
        ```

        Grating with reflections:

        ```python
        s_matrix = sax.models.grating_coupler(
            wl=1.55,
            loss=3.5,
            reflection=0.05,  # 5% waveguide reflection
            reflection_fiber=0.02,  # 2% fiber reflection
        )
        waveguide_reflection = abs(s_matrix[("in0", "in0")]) ** 2
        fiber_reflection = abs(s_matrix[("out0", "out0")]) ** 2
        ```

    Note:
        The transmission profile follows a Gaussian shape:
        T(λ) = T₀ * exp(-((λ-λ₀)/σ)²)

        Where σ = bandwidth / (2*√(2*ln(2))) converts FWHM to Gaussian width.

        This model assumes:
        - Gaussian spectral response (typical for uniform gratings)
        - Wavelength-independent loss coefficient
        - Linear polarization (TE or TM)
        - Single-mode fiber coupling
        - No higher-order diffraction effects

        Real grating couplers may exhibit:
        - Non-Gaussian spectral shapes
        - Polarization dependence
        - Temperature sensitivity
        - Higher-order grating resonances
        - Angular sensitivity to fiber positioning

        For more accurate modeling, consider using measured spectral data
        or finite-element simulation results.

    ```
                      out0/o2
                       fiber
                   /  /  /  /
                  /  /  /  /

                _|-|_|-|_|-|__
        in0/o1                |
                 _____________|
    ```

    ```python
    # mkdocs: render
    import matplotlib.pyplot as plt
    import numpy as np
    import sax

    sax.set_port_naming_strategy("optical")

    wavelengths = np.linspace(1.5, 1.6, 101)
    s = sax.models.grating_coupler(
        wl=wavelengths,
        loss=3.0,
        bandwidth=0.035,
    )
    plt.figure()
    plt.plot(wavelengths, np.abs(s[("o1", "o2")]) ** 2, label="Transmission")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Power")
    plt.legend()
    ```
    """
    one = jnp.ones_like(wl)
    reflection = jnp.asarray(reflection) * one
    reflection_fiber = jnp.asarray(reflection_fiber) * one
    amplitude = jnp.asarray(10 ** (-loss / 20))
    sigma = jnp.asarray(bandwidth / (2 * jnp.sqrt(2 * jnp.log(2))))
    transmission = jnp.asarray(amplitude * jnp.exp(-((wl - wl0) ** 2) / (2 * sigma**2)))
    p = sax.PortNamer(1, 1)
    return sax.reciprocal(
        {
            (p.in0, p.in0): reflection,
            (p.in0, p.out0): transmission,
            (p.out0, p.in0): transmission,
            (p.out0, p.out0): reflection_fiber,
        }
    )
