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
    """Ideal 1x2 multimode interference (MMI) splitter model.

    This function models an ideal 1x2 MMI splitter that divides input power
    between two output ports. The model is implemented as an ideal splitter
    and represents the theoretical limit of MMI splitter performance.

    MMI devices use multimode interference in a wide waveguide to achieve
    power splitting through self-imaging effects. This ideal model assumes
    perfect operation without considering the detailed physics of multimode
    propagation.

    Args:
        coupling: Power coupling ratio between 0 and 1. Determines the fraction
            of input power that couples to the second output port (out1).
            - coupling=0.0: All power to out0
            - coupling=0.5: Equal power splitting (3dB splitter)
            - coupling=1.0: All power to out1
            Defaults to 0.5 for equal splitting.

    Returns:
        S-matrix dictionary representing the ideal MMI splitter behavior.

    Examples:
        Equal power MMI splitter:

        ```python
        import sax

        s_matrix = sax.models.mmi1x2_ideal(coupling=0.5)
        power_out0 = abs(s_matrix[("in0", "out0")]) ** 2
        power_out1 = abs(s_matrix[("in0", "out1")]) ** 2
        print(f"Output powers: {power_out0:.3f}, {power_out1:.3f}")
        ```

        Asymmetric MMI splitter:

        ```python
        s_matrix = sax.models.mmi1x2_ideal(coupling=0.25)  # 25% to out1
        power_out0 = abs(s_matrix[("in0", "out0")]) ** 2  # Should be 0.75
        power_out1 = abs(s_matrix[("in0", "out1")]) ** 2  # Should be 0.25
        ```

    Note:
        This is an idealized model that assumes:
        - Perfect power conservation
        - No wavelength dependence
        - No reflection or insertion loss
        - Ideal multimode interference

        Real MMI devices exhibit wavelength-dependent behavior, finite bandwidth,
        and may have imbalance between output ports. For more realistic modeling,
        use the dispersive mmi1x2() function.
    """
    return splitter_ideal(coupling=coupling)


@jax.jit
@validate_call
def mmi2x2_ideal(*, coupling: sax.FloatArrayLike = 0.5) -> sax.SDict:
    """Ideal 2x2 multimode interference (MMI) coupler model.

    This function models an ideal 2x2 MMI coupler that can function as either
    a directional coupler or a 90-degree hybrid depending on the design. The
    model is implemented as an ideal coupler and represents the theoretical
    limit of MMI coupler performance.

    MMI couplers use multimode interference effects in a wide waveguide to
    achieve controllable coupling between input and output ports. This ideal
    model assumes perfect operation without detailed multimode analysis.

    Args:
        coupling: Power coupling coefficient between 0 and 1. Determines the
            fraction of power that couples between the two arms.
            - coupling=0.0: No coupling (straight transmission)
            - coupling=0.5: 3dB coupler (equal bar/cross transmission)
            - coupling=1.0: Complete cross-coupling
            Defaults to 0.5.

    Returns:
        S-matrix dictionary representing the ideal MMI coupler behavior.

    Examples:
        3dB MMI coupler:

        ```python
        import sax

        s_matrix = sax.models.mmi2x2_ideal(coupling=0.5)
        bar_power = abs(s_matrix[("in0", "out0")]) ** 2  # Bar transmission
        cross_power = abs(s_matrix[("in0", "out1")]) ** 2  # Cross transmission
        print(f"Bar: {bar_power:.3f}, Cross: {cross_power:.3f}")
        ```

        Variable coupling MMI:

        ```python
        s_matrix = sax.models.mmi2x2_ideal(coupling=0.8)  # 80% coupling
        bar_power = abs(s_matrix[("in0", "out0")]) ** 2  # Should be 0.2
        cross_power = abs(s_matrix[("in0", "out1")]) ** 2  # Should be 0.8
        ```

    Note:
        This is an idealized model that assumes:
        - Perfect power conservation
        - Ideal 90-degree phase relationships
        - No wavelength dependence
        - No reflection or insertion loss

        Real MMI couplers have wavelength-dependent coupling, finite bandwidth,
        and may deviate from ideal phase relationships. For more realistic
        modeling, use the dispersive mmi2x2() function.
    """
    return coupler_ideal(coupling=coupling)


@jax.jit
@validate_call
def mmi1x2(
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    fwhm: sax.FloatArrayLike = 0.2,
    loss_dB: sax.FloatArrayLike = 0.3,
) -> sax.SDict:
    r"""Realistic 1x2 MMI splitter model with dispersion and loss.

    This function models a realistic 1x2 multimode interference (MMI) splitter
    with wavelength-dependent transmission, insertion loss, and finite bandwidth.
    The model uses a Gaussian spectral response to capture the typical behavior
    of fabricated MMI devices.

    The MMI splitter divides input power equally between two output ports at
    the design wavelength but exhibits wavelength-dependent imbalance and
    rolloff away from the center wavelength.

    Args:
        wl: Operating wavelength in micrometers. Can be a scalar or array for
            multi-wavelength simulations. Defaults to 1.55 μm.
        wl0: Center wavelength in micrometers where optimal splitting occurs.
            This is the design wavelength of the MMI device. Defaults to 1.55 μm.
        fwhm: Full width at half maximum bandwidth in micrometers. Determines
            the spectral width over which the MMI maintains good performance.
            Typical values: 0.1-0.3 μm. Defaults to 0.2 μm.
        loss_dB: Insertion loss in dB at the center wavelength. Includes
            scattering losses, mode conversion losses, and fabrication
            imperfections. Defaults to 0.3 dB.

    Returns:
        S-matrix dictionary representing the dispersive MMI splitter behavior.

    Examples:
        Basic MMI splitter:

        ```python
        import sax
        import numpy as np

        s_matrix = sax.models.mmi1x2(wl=1.55, fwhm=0.15, loss_dB=0.5)
        power_out0 = abs(s_matrix[("o1", "o2")]) ** 2
        power_out1 = abs(s_matrix[("o1", "o3")]) ** 2
        total_power = power_out0 + power_out1
        insertion_loss_dB = -10 * np.log10(total_power)
        ```

        Spectral analysis:

        ```python
        wavelengths = np.linspace(1.4, 1.7, 301)
        s_matrices = sax.models.mmi1x2(wl=wavelengths, wl0=1.55, fwhm=0.2, loss_dB=0.4)
        transmission = np.abs(s_matrices[("o1", "o2")]) ** 2
        # Analyze spectral response and bandwidth
        ```

        Bandwidth optimization:

        ```python
        # Design for specific bandwidth
        target_bandwidth = 0.1  # 100 nm
        s_matrix = sax.models.mmi1x2(wl=1.55, fwhm=target_bandwidth, loss_dB=0.3)
        ```

    Note:
        The spectral response follows a Gaussian profile in the frequency domain:
        T(λ) = T₀ * exp(-((f-f₀)/σ_f)²)

        Where f = 1/λ is the optical frequency and σ_f is related to the FWHM.

        This model assumes:
        - Equal splitting at the center wavelength
        - Gaussian spectral response (typical for well-designed MMIs)
        - Wavelength-independent loss coefficient
        - No reflection at input/output interfaces

        Real MMI devices may exhibit:
        - Wavelength-dependent splitting ratio imbalance
        - Higher-order spectral features
        - Polarization dependence
        - Temperature sensitivity
        - Phase errors between outputs

    ```
               length_mmi
                <------>
                ________
               |        |
               |         \__
               |          __  o2
            __/          /_ _ _ _
         o1 __          | _ _ _ _| gap_mmi
              \          \__
               |          __  o3
               |         /
               |________|

             <->
        length_taper
    ```

    ```python
    # mkdocs: render
    import matplotlib.pyplot as plt
    import numpy as np
    import sax

    sax.set_port_naming_strategy("optical")

    wavelengths = np.linspace(1.5, 1.6, 101)
    s = sax.models.mmi1x2(wl=wavelengths, fwhm=0.15, loss_dB=0.5)
    transmission_o2 = np.abs(s[("o1", "o2")]) ** 2
    transmission_o3 = np.abs(s[("o1", "o3")]) ** 2
    plt.plot(wavelengths, transmission_o2, label="Output 1")
    plt.plot(wavelengths, transmission_o3, label="Output 2")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Transmission")
    plt.legend()
    ```

    """
    thru = _mmi_amp(wl=wl, wl0=wl0, fwhm=fwhm, loss_dB=loss_dB) / 2**0.5

    p = sax.PortNamer(1, 2)
    return sax.reciprocal(
        {
            (p.o1, p.o2): thru,
            (p.o1, p.o3): thru,
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
    r"""Realistic 2x2 MMI coupler model with dispersion and asymmetry.

    This function models a realistic 2x2 multimode interference (MMI) coupler
    with wavelength-dependent behavior, insertion loss, and the ability to
    model fabrication-induced asymmetries between bar and cross ports.

    The MMI coupler can function as a directional coupler, 90-degree hybrid,
    or beam splitter depending on the design parameters. This model includes
    separate control over bar and cross port characteristics.

    Args:
        wl: Operating wavelength in micrometers. Can be a scalar or array for
            multi-wavelength simulations. Defaults to 1.55 μm.
        wl0: Center wavelength in micrometers for optimal coupling performance.
            This is the design wavelength of the MMI device. Defaults to 1.55 μm.
        fwhm: Full width at half maximum bandwidth in micrometers. Determines
            the spectral width of good performance. Defaults to 0.2 μm.
        loss_dB: Base insertion loss in dB at the center wavelength. Used for
            both ports unless overridden. Defaults to 0.3 dB.
        shift: Wavelength shift in micrometers applied to both bar and cross
            responses. Models fabrication variations. Defaults to 0.005 μm.
        loss_dB_cross: Optional separate insertion loss in dB for cross ports.
            If None, uses loss_dB. Allows modeling of asymmetric loss.
        loss_dB_thru: Optional separate insertion loss in dB for bar (through)
            ports. If None, uses loss_dB. Allows modeling of asymmetric loss.
        splitting_ratio_cross: Power splitting ratio for cross ports (0 to 1).
            Allows modeling of imbalanced coupling. Defaults to 0.5.
        splitting_ratio_thru: Power splitting ratio for bar ports (0 to 1).
            Allows modeling of imbalanced transmission. Defaults to 0.5.

    Returns:
        S-matrix dictionary representing the realistic MMI coupler behavior.

    Examples:
        Symmetric 3dB MMI coupler:

        ```python
        import sax
        import numpy as np

        s_matrix = sax.models.mmi2x2(
            wl=1.55,
            fwhm=0.15,
            loss_dB=0.4,
            splitting_ratio_cross=0.5,
            splitting_ratio_thru=0.5,
        )
        bar_power = abs(s_matrix[("o1", "o3")]) ** 2
        cross_power = abs(s_matrix[("o1", "o4")]) ** 2
        ```

        Asymmetric MMI with different bar/cross losses:

        ```python
        s_matrix = sax.models.mmi2x2(
            wl=1.55,
            loss_dB_thru=0.3,  # Lower bar loss
            loss_dB_cross=0.6,  # Higher cross loss
            splitting_ratio_cross=0.4,  # Imbalanced coupling
            splitting_ratio_thru=0.6,
        )
        ```

        Wavelength-shifted MMI (fabrication variation):

        ```python
        s_matrix = sax.models.mmi2x2(
            wl=1.55,
            wl0=1.55,
            shift=0.01,  # 10 nm shift from process variation
            fwhm=0.12,
        )
        ```

        Spectral analysis of MMI coupler:

        ```python
        wavelengths = np.linspace(1.45, 1.65, 201)
        s_matrices = sax.models.mmi2x2(wl=wavelengths, wl0=1.55, fwhm=0.18, loss_dB=0.5)
        bar_transmission = np.abs(s_matrices[("o1", "o3")]) ** 2
        cross_transmission = np.abs(s_matrices[("o1", "o4")]) ** 2
        ```

    Note:
        The cross-coupled terms include a 90-degree phase shift (1j factor)
        to maintain proper S-matrix properties and represent the physical
        phase relationship in MMI devices.

        This model includes:
        - Separate spectral responses for bar and cross ports
        - Asymmetric loss modeling
        - Wavelength shift for process variations
        - Independent splitting ratio control

        The model assumes:
        - Gaussian spectral responses
        - Linear amplitude relationships
        - Reciprocal device behavior
        - No higher-order spectral features

        Real MMI devices may exhibit:
        - Non-Gaussian spectral shapes
        - Polarization dependence
        - Temperature sensitivity
        - Multimode interference patterns
        - Phase imbalance between outputs

    ```
               length_mmi
                <------>
                ________
               |        |
            __/          \__
        o2  __            __  o3
              \          /_ _ _ _
              |         | _ _ _ _| gap_output_tapers
            __/          \__
        o1  __            __  o4
              \          /
               |________|
             | |
             <->
        length_taper
    ```

    ```python
    # mkdocs: render
    import matplotlib.pyplot as plt
    import numpy as np
    import sax

    sax.set_port_naming_strategy("optical")

    wavelengths = np.linspace(1.5, 1.6, 101)
    s = sax.models.mmi2x2(wl=wavelengths, fwhm=0.15, loss_dB=0.5)
    bar_transmission = np.abs(s[("o1", "o3")]) ** 2
    cross_transmission = np.abs(s[("o1", "o4")]) ** 2
    plt.plot(wavelengths, bar_transmission, label="Bar")
    plt.plot(wavelengths, cross_transmission, label="Cross")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Transmission")
    plt.legend()
    ```
    """
    loss_dB_cross = loss_dB_cross or loss_dB
    loss_dB_thru = loss_dB_thru or loss_dB

    # Convert splitting ratios from power to amplitude by taking the square root
    amplitude_ratio_thru = splitting_ratio_thru**0.5
    amplitude_ratio_cross = splitting_ratio_cross**0.5

    # _mmi_amp already includes the loss, so we don't need to apply it again
    thru = (
        _mmi_amp(wl=wl, wl0=wl0 + shift, fwhm=fwhm, loss_dB=loss_dB_thru)
        * amplitude_ratio_thru
    )
    cross = (
        1j
        * _mmi_amp(wl=wl, wl0=wl0 + shift, fwhm=fwhm, loss_dB=loss_dB_cross)
        * amplitude_ratio_cross
    )

    p = sax.PortNamer(2, 2)
    return sax.reciprocal(
        {
            (p.o1, p.o3): thru,
            (p.o1, p.o4): cross,
            (p.o2, p.o3): cross,
            (p.o2, p.o4): thru,
        }
    )


def _mmi_amp(
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    fwhm: sax.FloatArrayLike = 0.2,
    loss_dB: sax.FloatArrayLike = 0.3,
) -> sax.FloatArray:
    """Calculate MMI amplitude response with Gaussian spectral profile.

    This helper function computes the amplitude transmission coefficient for
    MMI devices using a Gaussian spectral response in the frequency domain.
    The model converts between wavelength and frequency domains to properly
    capture the physical behavior of multimode interference.

    Args:
        wl: Operating wavelength in micrometers.
        wl0: Center wavelength in micrometers.
        fwhm: Full width at half maximum bandwidth in micrometers.
        loss_dB: Peak insertion loss in dB at the center wavelength.

    Returns:
        The mmi amplitude

    Note:
        The function works in the frequency domain (f = 1/λ) to ensure
        symmetric Gaussian responses, then converts back to wavelength domain.
        The amplitude is the square root of the power transmission to maintain
        proper S-matrix scaling.
    """
    # Convert loss from dB to amplitude directly (not power)
    max_amplitude = 10 ** (-abs(loss_dB) / 20)
    f = 1 / wl
    f0 = 1 / wl0
    f1 = 1 / (wl0 + fwhm / 2)
    f2 = 1 / (wl0 - fwhm / 2)
    _fwhm = f2 - f1

    sigma = _fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    # Gaussian response in frequency domain
    spectral_response = jnp.exp(-((f - f0) ** 2) / (2 * sigma**2))
    # Apply loss to amplitude, not power
    amplitude = max_amplitude * spectral_response / spectral_response.max()
    return jnp.asarray(amplitude)


def _mmi_nxn(
    n: int,
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    fwhm: sax.FloatArrayLike = 0.2,
    loss_dB: sax.FloatArrayLike | None = None,
    shift: sax.FloatArrayLike | None = None,
    splitting_matrix: sax.FloatArray2D | None = None,
) -> sax.SDict:
    """General n x n MMI model with configurable splitting matrix.

    This function provides a flexible framework for modeling arbitrary nxn
    MMI devices with custom power splitting matrices, wavelength-dependent
    responses, and port-specific loss and wavelength shift parameters.

    Args:
        n: Number of input and output ports (nxn MMI device).
        wl: Operating wavelength in micrometers. Defaults to 1.55 μm.
        wl0: Center wavelength in micrometers. Defaults to 1.55 μm.
        fwhm: Full width at half maximum bandwidth in micrometers. Defaults to 0.2 μm.
        loss_dB: Array of loss values in dB for each output port. If None,
            assumes lossless operation. Can be a scalar or array of length n.
        shift: Array of wavelength shifts in micrometers for each output port.
            If None, no shifts are applied. Can be a scalar or array of length n.
        splitting_matrix: nxn matrix defining power splitting ratios between
            ports. Element [i,j] is the power fraction from input i to output j.
            If None, assumes uniform splitting (1/n to each output).

    Returns:
        The mmi amplitude

    Examples:
        4x4 MMI with custom splitting:

        ```python
        import numpy as np
        import sax

        # Create custom 4x4 splitting matrix
        splitting = np.array(
            [
                [0.7, 0.1, 0.1, 0.1],  # Input 0 distribution
                [0.1, 0.7, 0.1, 0.1],  # Input 1 distribution
                [0.1, 0.1, 0.7, 0.1],  # Input 2 distribution
                [0.1, 0.1, 0.1, 0.7],  # Input 3 distribution
            ]
        )
        s_matrix = sax.models._mmi_nxn(n=4, wl=1.55, splitting_matrix=splitting)
        ```

        8x8 MMI with port-specific losses:

        ```python
        losses = np.array([0.2, 0.3, 0.25, 0.35, 0.3, 0.4, 0.3, 0.2])
        s_matrix = sax.models._mmi_nxn(n=8, loss_dB=losses, fwhm=0.15)
        ```

    Note:
        This is a generalized model that can represent various MMI configurations:
        - Star couplers (nxn with uniform splitting)
        - Asymmetric splitters/combiners
        - Wavelength-selective devices
        - Process-variation modeling

        The splitting matrix should satisfy power conservation constraints
        for physical devices (row sums ≤ 1). The model does not enforce
        this constraint to allow modeling of lossy or gain devices.
    """
    _loss_dB = jnp.zeros(n) if loss_dB is None else jnp.asarray(loss_dB)
    _shift = jnp.zeros(n) if shift is None else jnp.asarray(shift)
    _splitting_matrix = (
        jnp.full((n, n), 1 / n)
        if splitting_matrix is None
        else jnp.asarray(splitting_matrix)
    )

    S = {}
    p = sax.PortNamer(n, n)
    for i in range(n):
        for j in range(n):
            amplitude = _mmi_amp(wl, wl0 + _shift[j], fwhm, _loss_dB[j])
            amplitude *= jnp.sqrt(_splitting_matrix[i][j])
            # _mmi_amp already includes the loss, so no additional loss factor needed
            S[(p[i], p[n + j])] = amplitude

    return sax.reciprocal(S)
