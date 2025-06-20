"""SAX Default Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def straight(
    *,
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    neff: sax.FloatArrayLike = 2.34,
    ng: sax.FloatArrayLike = 3.4,
    length: sax.FloatArrayLike = 10.0,
    loss_dB_cm: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    """Dispersive straight waveguide model.

    This function models a straight waveguide section with chromatic dispersion
    and propagation loss. The model includes both material and waveguide dispersion
    effects through the group index parameter and supports wavelength-dependent
    loss characteristics.

    The model calculates:
    - Wavelength-dependent effective index using group index
    - Phase accumulation due to propagation
    - Power attenuation due to loss mechanisms
    - Bidirectional transmission with reciprocal behavior

    Args:
        wl: Operating wavelength in micrometers. Can be a scalar or array for
            multi-wavelength simulations. Defaults to 1.55 μm.
        wl0: Reference wavelength in micrometers used for dispersion calculation.
            This is typically the design wavelength where neff is specified.
            Defaults to 1.55 μm.
        neff: Effective refractive index at the reference wavelength. This value
            represents the fundamental mode effective index and determines the
            phase velocity. Defaults to 2.34 (typical for silicon).
        ng: Group refractive index at the reference wavelength. Used to calculate
            chromatic dispersion: ng = neff - lambda * d(neff)/d(lambda).
            Typically ng > neff for normal dispersion. Defaults to 3.4.
        length: Physical length of the waveguide in micrometers. Determines both
            the total phase accumulation and loss. Defaults to 10.0 μm.
        loss_dB_cm: Propagation loss in dB/cm. Includes material absorption,
            scattering losses, and other loss mechanisms. Set to 0.0 for
            lossless modeling. Defaults to 0.0 dB/cm.

    Returns:
        S-matrix dictionary containing the complex transmission coefficient.

    Examples:
        Basic lossless waveguide:

        ```python
        import sax

        # Lossless silicon waveguide
        s_matrix = sax.models.straight(
            wl=1.55, length=100.0, neff=2.4, ng=3.8, loss_dB_cm=0.0
        )
        phase = jnp.angle(s_matrix[("in0", "out0")])
        print(f"Phase shift: {phase:.3f} radians")
        ```

        Lossy waveguide analysis:

        ```python
        # Waveguide with realistic loss
        s_matrix = sax.models.straight(
            wl=1.55,
            length=1000.0,  # 1 mm
            loss_dB_cm=2.0,  # 2 dB/cm loss
        )
        transmission = abs(s_matrix[("in0", "out0")]) ** 2
        loss_dB = -10 * jnp.log10(transmission)
        print(f"Total loss: {loss_dB:.2f} dB")
        ```

        Dispersion analysis:

        ```python
        import numpy as np

        wavelengths = np.linspace(1.5, 1.6, 101)
        s_matrices = sax.models.straight(
            wl=wavelengths,
            length=100.0,
            neff=2.4,
            ng=4.0,  # High dispersion
        )
        phases = np.angle(s_matrices[("in0", "out0")])
        group_delay = np.gradient(phases, wavelengths)
        ```

    Note:
        The dispersion model uses a linear approximation:
        neff(λ) = neff(λ0) - (λ - λ0) * (ng - neff) / λ0

        This is accurate for small wavelength deviations but may need higher-order
        terms for broadband applications. The model assumes:
        - Single-mode propagation
        - Linear dispersion (first-order only)
        - Wavelength-independent loss coefficient
        - Perfect reciprocity
    """
    dwl: sax.FloatArray = sax.into[sax.FloatArray](wl) - wl0
    dneff_dwl = (ng - neff) / wl0
    _neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * _neff * length / wl
    amplitude = jnp.asarray(10 ** (-1e-4 * loss_dB_cm * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    p = sax.PortNamer(1, 1)
    return sax.reciprocal(
        {
            (p.in0, p.out0): transmission,
        },
    )


@jax.jit
@validate_call
def attenuator(*, loss: sax.FloatArrayLike = 0.0) -> sax.SDict:
    """Simple optical attenuator model.

    This function models a variable optical attenuator (VOA) that provides
    controllable attenuation without phase change. The device reduces optical
    power by a specified amount while maintaining the phase of the transmitted
    light.

    The attenuator is reciprocal and lossless from an S-matrix perspective,
    meaning the specified loss represents intentional signal reduction rather
    than parasitic loss that would break unitarity.

    Args:
        loss: Attenuation in decibels (dB). Positive values represent attenuation:
            - loss = 0.0: No attenuation (unity transmission)
            - loss = 3.0: 3 dB attenuation (50% power transmission)
            - loss = 10.0: 10 dB attenuation (10% power transmission)
            Can be wavelength-dependent for spectral filtering applications.
            Defaults to 0.0 dB.

    Returns:
        S-matrix dictionary containing the complex transmission coefficient.

    Examples:
        Fixed attenuator:

        ```python
        import sax

        # 6 dB attenuator
        s_matrix = sax.models.attenuator(loss=6.0)
        transmission = abs(s_matrix[("in0", "out0")]) ** 2
        print(f"Power transmission: {transmission:.3f}")  # Should be ~0.251

        # Verify in dB
        loss_dB = -10 * jnp.log10(transmission)
        print(f"Loss: {loss_dB:.1f} dB")  # Should be 6.0 dB
        ```

        Variable attenuator:

        ```python
        import numpy as np

        losses = np.linspace(0, 20, 101)  # 0 to 20 dB
        transmissions = []
        for loss_val in losses:
            s = sax.models.attenuator(loss=loss_val)
            transmissions.append(abs(s[("in0", "out0")]) ** 2)
        ```

        Wavelength-dependent attenuator (filter):

        ```python
        wavelengths = np.linspace(1.5, 1.6, 101)
        # Gaussian spectral filter
        center_wl = 1.55
        bandwidth = 0.01
        spectral_loss = 20 * np.exp(-(((wavelengths - center_wl) / bandwidth) ** 2))
        s_matrices = sax.models.attenuator(loss=spectral_loss)
        ```

    Note:
        This model represents an ideal attenuator with:
        - No wavelength dependence (unless explicitly provided)
        - No reflection at input or output
        - No phase change
        - Perfect reciprocity

        Real attenuators may exhibit:
        - Wavelength-dependent loss variation
        - Small amounts of reflection
        - Polarization-dependent loss (PDL)
        - Temperature sensitivity

        For more complex filtering behavior, consider combining multiple
        components or using specialized filter models.
    """
    transmission = jnp.asarray(10 ** (-loss / 20), dtype=complex)
    p = sax.PortNamer(1, 1)
    return sax.reciprocal(
        {
            (p.in0, p.out0): transmission,
        }
    )


@jax.jit
@validate_call
def phase_shifter(
    wl: sax.FloatArrayLike = 1.55,
    neff: sax.FloatArrayLike = 2.34,
    voltage: sax.FloatArrayLike = 0,
    length: sax.FloatArrayLike = 10,
    loss: sax.FloatArrayLike = 0.0,
) -> sax.SDict:
    """Simple voltage-controlled phase shifter model.

    This function models a thermo-optic or electro-optic phase shifter that
    provides voltage-controlled phase modulation. The device introduces a
    voltage-dependent phase shift while maintaining (nearly) constant amplitude.

    The model combines:
    - Passive waveguide propagation phase
    - Voltage-induced phase shift (linear relationship)
    - Optional loss from the active region

    Args:
        wl: Operating wavelength in micrometers. The phase shift has a 1/λ
            dependence from the propagation term. Defaults to 1.55 μm.
        neff: Effective refractive index of the unperturbed waveguide mode.
            This determines the baseline propagation phase. Defaults to 2.34.
        voltage: Applied voltage in volts. The phase shift is assumed to be
            linearly proportional to voltage with a coefficient of π rad/V.
            Positive voltage increases the phase. Defaults to 0 V.
        length: Active length of the phase shifter in micrometers. Both the
            propagation phase and voltage-induced phase scale with length.
            Defaults to 10 μm.
        loss: Additional loss in dB introduced by the active region (e.g.,
            free-carrier absorption in silicon modulators). This loss is
            proportional to the device length. Defaults to 0.0 dB.

    Returns:
        S-matrix dictionary containing the complex-valued transmission coefficient.

    Examples:
        Basic phase shifter:

        ```python
        import sax
        import numpy as np

        # No voltage applied
        s_matrix = sax.models.phase_shifter(wl=1.55, voltage=0.0, length=100.0)
        phase_0V = np.angle(s_matrix[("o1", "o2")])

        # 1V applied
        s_matrix = sax.models.phase_shifter(wl=1.55, voltage=1.0, length=100.0)
        phase_1V = np.angle(s_matrix[("o1", "o2")])
        voltage_phase_shift = phase_1V - phase_0V
        print(f"Voltage-induced phase shift: {voltage_phase_shift:.3f} rad")
        ```

        Phase modulation analysis:

        ```python
        voltages = np.linspace(-2, 2, 101)
        phases = []
        for v in voltages:
            s = sax.models.phase_shifter(voltage=v, length=50.0)
            phases.append(np.angle(s[("o1", "o2")]))
        phases = np.array(phases)
        # Should see linear relationship with slope = π * length
        ```

        Lossy phase shifter (e.g., silicon modulator):

        ```python
        s_matrix = sax.models.phase_shifter(
            voltage=1.0,
            length=1000.0,  # 1 mm
            loss=0.5,  # 0.5 dB loss
        )
        transmission = abs(s_matrix[("o1", "o2")]) ** 2
        phase_shift = np.angle(s_matrix[("o1", "o2")])
        ```

    Note:
        This simplified model assumes:
        - Linear voltage-to-phase relationship (π rad/V)
        - Negligible voltage dependence of loss
        - No reflection from the active region
        - Perfect reciprocity

        Real phase shifters may exhibit:
        - Nonlinear voltage response
        - Voltage-dependent loss
        - Bandwidth limitations
        - Temperature sensitivity
        - Polarization dependence

        The model uses a simple π rad/V coefficient which may need calibration
        for specific devices and operating conditions.
    """
    deltaphi = voltage * jnp.pi
    phase = 2 * jnp.pi * neff * length / wl + deltaphi
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    p = sax.PortNamer(1, 1)
    return sax.reciprocal(
        {
            (p.o1, p.o2): transmission,
        }
    )
