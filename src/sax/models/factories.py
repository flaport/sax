"""SAX Default Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@validate_call
def model_2port(p1: sax.Name, p2: sax.Name) -> sax.SDictModel:
    """Generate a general 2-port model factory with custom port names.

    This function creates a factory that generates simple 2-port devices with
    unity transmission between the specified ports. The resulting model provides
    ideal transmission with no loss, reflection, or wavelength dependence.

    This factory is useful for creating custom 2-port models with specific
    port naming requirements or as a starting point for more complex models.

    Args:
        p1: Name of the first port (typically input).
        p2: Name of the second port (typically output).

    Returns:
        A 2-port model

    Examples:
        Create a custom waveguide model:

        ```python
        import sax

        # Create a 2-port model with custom port names
        waveguide_model = sax.models.model_2port("input", "output")
        s_matrix = waveguide_model(wl=1.55)
        print(s_matrix[("input", "output")])  # Should be 1.0
        ```

        Create multiple models with different port names:

        ```python
        isolator_model = sax.models.model_2port("in", "out")
        delay_line_model = sax.models.model_2port("start", "end")

        # Use in circuit simulations
        s1 = isolator_model(wl=1.55)
        s2 = delay_line_model(wl=1.55)
        ```

    Note:
        The generated model assumes:
        - Perfect unity transmission (no loss)
        - No wavelength dependence
        - No reflection at either port
        - Bidirectional reciprocity

        For more complex behavior, consider using the specific component
        models (straight, attenuator, phase_shifter) or implementing
        custom models with proper physical parameters.
    """

    @jax.jit
    @validate_call
    def model_2port(wl: sax.FloatArrayLike = 1.5) -> sax.SDict:
        wl = jnp.asarray(wl)
        return sax.reciprocal({(p1, p2): jnp.ones_like(wl)})

    return model_2port


@validate_call
def model_3port(p1: sax.Name, p2: sax.Name, p3: sax.Name) -> sax.SDictModel:
    """Generate a general 3-port model factory with custom port names.

    This function creates a factory that generates 3-port devices implementing
    a 1-to-2 power splitter with equal splitting ratio. The model provides
    ideal 3dB splitting from the first port to the other two ports.

    This factory is useful for creating custom splitter models with specific
    port naming or as a foundation for more complex 3-port devices.

    Args:
        p1: Name of the input port.
        p2: Name of the first output port.
        p3: Name of the second output port.

    Returns:
        A 3-port model

    Examples:
        Create a custom Y-branch splitter:

        ```python
        import sax

        y_branch_model = sax.models.model_3port("input", "branch1", "branch2")
        s_matrix = y_branch_model(wl=1.55)

        # Check equal splitting (3dB each)
        power_branch1 = abs(s_matrix[("input", "branch1")]) ** 2
        power_branch2 = abs(s_matrix[("input", "branch2")]) ** 2
        print(f"Powers: {power_branch1:.3f}, {power_branch2:.3f}")  # Both 0.5
        ```

        Create a tap coupler model:

        ```python
        tap_model = sax.models.model_3port("in", "thru", "drop")
        s_matrix = tap_model(wl=1.55)
        # Equal 50/50 tapping
        ```

        Multi-wavelength simulation:

        ```python
        import numpy as np

        wavelengths = np.linspace(1.5, 1.6, 101)
        splitter_model = sax.models.model_3port("in", "out1", "out2")
        s_matrices = splitter_model(wl=wavelengths)
        # Wavelength-independent equal splitting
        ```

    Note:
        The generated model implements:
        - Equal 3dB power splitting (50/50)
        - Unity amplitude coefficients of 1/√2
        - No wavelength dependence
        - Perfect reciprocity
        - No reflection at any port

        For wavelength-dependent or asymmetric splitting, consider using
        the splitter_ideal model with custom coupling ratios or implementing
        dispersive splitter models.
    """

    @jax.jit
    @validate_call
    def model_3port(wl: sax.FloatArrayLike = 1.5) -> sax.SDict:
        wl = jnp.asarray(wl)
        thru = jnp.ones_like(wl) / jnp.sqrt(2)
        return sax.reciprocal(
            {
                (p1, p2): thru,
                (p1, p3): thru,
            }
        )

    return model_3port


@validate_call
def model_4port(
    p1: sax.Name, p2: sax.Name, p3: sax.Name, p4: sax.Name
) -> sax.SDictModel:
    """Generate a general 4-port model factory with custom port names.

    This function creates a factory that generates 4-port devices implementing
    a 2x2 directional coupler with 3dB coupling. The model provides ideal
    coupling behavior with proper 90-degree phase relationships.

    The coupling configuration is:
    - Bar transmission: p1↔p4, p2↔p3 with amplitude 1/√2
    - Cross coupling: p1↔p3, p2↔p4 with amplitude j/√2 (90° phase shift)

    Args:
        p1: Name of the first input port.
        p2: Name of the second input port.
        p3: Name of the first output port.
        p4: Name of the second output port.

    Returns:
        A 4-port model

    Examples:
        Create a custom directional coupler:

        ```python
        import sax

        dc_model = sax.models.model_4port("in1", "in2", "out1", "out2")
        s_matrix = dc_model(wl=1.55)

        # Check 3dB coupling
        bar_power = abs(s_matrix[("in1", "out2")]) ** 2
        cross_power = abs(s_matrix[("in1", "out1")]) ** 2
        print(f"Bar: {bar_power:.3f}, Cross: {cross_power:.3f}")  # Both 0.5

        # Check 90-degree phase relationship
        bar_phase = jnp.angle(s_matrix[("in1", "out2")])
        cross_phase = jnp.angle(s_matrix[("in1", "out1")])
        phase_diff = cross_phase - bar_phase
        print(f"Phase difference: {phase_diff:.3f} rad")  # Should be π/2
        ```

        Create a 2x2 MMI model:

        ```python
        mmi_model = sax.models.model_4port("i1", "i2", "o1", "o2")
        s_matrix = mmi_model(wl=1.55)
        ```

        Multi-wavelength coupler analysis:

        ```python
        import numpy as np

        wavelengths = np.linspace(1.5, 1.6, 101)
        coupler_model = sax.models.model_4port("p1", "p2", "p3", "p4")
        s_matrices = coupler_model(wl=wavelengths)
        # Wavelength-independent 3dB coupling
        ```

    Note:
        The generated model implements:
        - 3dB (50/50) power coupling
        - 90-degree phase shift for cross-coupled terms (1j factor)
        - Perfect power conservation (unitary S-matrix)
        - No wavelength dependence
        - Complete reciprocity

        The S-matrix structure follows the standard directional coupler form:
        - Through paths: Real amplitude coefficients
        - Cross paths: Imaginary amplitude coefficients (j/√2)

        For variable coupling ratios or dispersive behavior, use the
        coupler_ideal or coupler models with appropriate parameters.
    """

    @jax.jit
    @validate_call
    def model_4port(wl: sax.FloatArrayLike = 1.5) -> sax.SDict:
        wl = jnp.asarray(wl)
        thru = jnp.ones_like(wl) / jnp.sqrt(2)
        cross = 1j * thru
        return sax.reciprocal(
            {
                (p1, p4): thru,
                (p2, p3): thru,
                (p1, p3): cross,
                (p2, p4): cross,
            }
        )

    return model_4port


@validate_call
def unitary(
    num_inputs: int,
    num_outputs: int,
    *,
    reciprocal: bool = True,
    diagonal: bool = False,
) -> sax.SCooModel:
    """Generate a unitary N×M optical device model.

    This function creates a model for a unitary optical device that conserves
    power while providing controllable coupling between input and output ports.
    The model generates a physically realizable S-matrix through singular value
    decomposition and proper normalization.

    Unitary devices are fundamental building blocks in photonics, including
    star couplers, multimode interference devices, and arbitrary unitary
    transformations for quantum photonic circuits.

    Args:
        num_inputs: Number of input ports for the device.
        num_outputs: Number of output ports for the device.
        reciprocal: If True, the device exhibits reciprocal behavior (S = S^T).
            This is typical for passive optical devices. Defaults to True.
        diagonal: If True, creates a diagonal coupling matrix (each input
            couples to only one output). If False, creates full coupling
            between all input-output pairs. Defaults to False.

    Returns:
        A unitary model

    Examples:
        Create a 4×4 star coupler:

        ```python
        import sax

        star_coupler = sax.models.unitary(4, 4, reciprocal=True, diagonal=False)
        Si, Sj, Sx, port_map = star_coupler(wl=1.55)
        # Full 4×4 unitary matrix with all-to-all coupling
        ```

        Create a 2×8 splitter array:

        ```python
        splitter_array = sax.models.unitary(2, 8, reciprocal=True)
        Si, Sj, Sx, port_map = splitter_array(wl=1.55)
        # Each input couples to all 8 outputs
        ```

        Create diagonal routing device:

        ```python
        router = sax.models.unitary(4, 4, diagonal=True, reciprocal=True)
        Si, Sj, Sx, port_map = router(wl=1.55)
        # Each input couples to only one output (permutation matrix)
        ```

        Non-reciprocal device (e.g., isolator array):

        ```python
        isolator_array = sax.models.unitary(3, 3, reciprocal=False)
        Si, Sj, Sx, port_map = isolator_array(wl=1.55)
        # Asymmetric transmission characteristics
        ```

    Note:
        The algorithm works by:
        1. Creating an initial coupling matrix based on the diagonal flag
        2. Applying SVD to ensure unitary properties: S = U @ diag(s) @ V†
        3. Normalizing to ensure power conservation
        4. Extracting the specified input/output submatrix

        The resulting S-matrix satisfies:
        - Power conservation: S† @ S = I (for square matrices)
        - Unitarity within the specified dimensions
        - Proper reciprocity if requested

        This model is ideal for:
        - Prototyping arbitrary unitary transformations
        - Modeling star couplers and fan-out devices
        - Quantum photonic circuit elements
        - Network analysis of complex optical systems

        For specific device physics, consider using dedicated models
        like MMI, coupler, or custom implementations with proper
        material and geometric parameters.
    """
    # let's create the squared S-matrix:
    N = max(num_inputs, num_outputs)
    S = jnp.zeros((2 * N, 2 * N), dtype=float)

    if not diagonal:
        S = S.at[:N, N:].set(1)
    else:
        r = jnp.arange(
            N,
            dtype=int,
        )  # reciprocal only works if num_inputs == num_outputs!
        S = S.at[r, N + r].set(1)

    if reciprocal:
        if not diagonal:
            S = S.at[N:, :N].set(1)
        else:
            r = jnp.arange(
                N,
                dtype=int,
            )  # reciprocal only works if num_inputs == num_outputs!
            S = S.at[N + r, r].set(1)

    # Now we need to normalize the squared S-matrix
    U, s, V = jnp.linalg.svd(S, full_matrices=False)
    S = jnp.sqrt(U @ jnp.diag(jnp.where(s > sax.EPS, 1, 0)) @ V)

    # Now create subset of this matrix we're interested in:
    r = jnp.concatenate(
        [jnp.arange(num_inputs, dtype=int), N + jnp.arange(num_outputs, dtype=int)],
        0,
    )
    S = S[r, :][:, r]

    # let's convert it in SCOO format:
    Si, Sj = jnp.where(S > sax.EPS)
    Sx = S[Si, Sj]

    # the last missing piece is a port map:
    p = sax.PortNamer(num_inputs, num_outputs)
    pm = {
        **{p[i]: i for i in range(num_inputs)},
        **{p[i + num_inputs]: i + num_inputs for i in range(num_outputs)},
    }

    @validate_call
    def func(*, wl: sax.FloatArrayLike = 1.5) -> sax.SCoo:
        wl_ = jnp.asarray(wl)
        Sx_ = jnp.broadcast_to(Sx, (*wl_.shape, *Sx.shape))
        return Si, Sj, Sx_, pm

    func.__name__ = f"unitary_{num_inputs}_{num_outputs}"
    func.__qualname__ = f"unitary_{num_inputs}_{num_outputs}"
    return jax.jit(func)


@validate_call
def copier(
    num_inputs: int,
    num_outputs: int,
    *,
    reciprocal: bool = True,
    diagonal: bool = False,
) -> sax.SCooModel:
    """Generate a power copying/amplification device model.

    This function creates a model for optical devices that can copy or amplify
    input signals to multiple outputs. Unlike unitary devices, copiers can
    provide gain and may not conserve power, making them useful for modeling
    amplifiers, laser arrays, or ideal copying devices.

    The copier model creates controlled coupling between inputs and outputs
    without the strict unitarity constraints of passive devices, allowing
    for flexible gain and splitting characteristics.

    Args:
        num_inputs: Number of input ports for the device.
        num_outputs: Number of output ports for the device.
        reciprocal: If True, the device exhibits reciprocal behavior where
            forward and reverse transmissions are equal. Defaults to True.
        diagonal: If True, creates diagonal coupling (each input couples to
            only one output). If False, creates full coupling between all
            input-output pairs. Defaults to False.

    Returns:
        A copier model

    Examples:
        Create a 1×4 optical amplifier/splitter:

        ```python
        import sax

        amp_splitter = sax.models.copier(1, 4, reciprocal=False)
        Si, Sj, Sx, port_map = amp_splitter(wl=1.55)
        # Single input amplified and split to 4 outputs
        ```

        Create a 2×2 bidirectional amplifier:

        ```python
        bidir_amp = sax.models.copier(2, 2, reciprocal=True, diagonal=True)
        Si, Sj, Sx, port_map = bidir_amp(wl=1.55)
        # Each input amplified to corresponding output
        ```

        Create a broadcast network:

        ```python
        broadcaster = sax.models.copier(3, 6, reciprocal=False)
        Si, Sj, Sx, port_map = broadcaster(wl=1.55)
        # Each input broadcast to all outputs with gain
        ```

        Multi-wavelength gain device:

        ```python
        import numpy as np

        wavelengths = np.linspace(1.5, 1.6, 101)
        gain_device = sax.models.copier(1, 2, diagonal=False)
        Si, Sj, Sx, port_map = gain_device(wl=wavelengths)
        # Wavelength-independent copying/amplification
        ```

    Note:
        The copier model differs from unitary devices in several key ways:

        1. Power Conservation: May not conserve power (can provide gain)
        2. Unitarity: S-matrix may not be unitary (|S|² can exceed 1)
        3. Physical Realizability: Requires active elements or non-reciprocal materials

        Applications include:
        - Optical amplifier modeling (SOAs, EDFAs, Raman)
        - Laser array modeling
        - Signal distribution networks
        - Quantum information processing (ideal copying devices)
        - Gain medium characterization

        The coupling matrix construction follows similar principles to the
        unitary model but without SVD normalization, allowing for arbitrary
        transmission coefficients.

        For realistic amplifier modeling, consider adding:
        - Wavelength-dependent gain profiles
        - Noise figures and ASE generation
        - Saturation effects
        - Polarization dependencies

        Note: True quantum copying of unknown states is forbidden by the
        no-cloning theorem, but this model can represent classical optical
        copying or quantum copying of known states.
    """
    # let's create the squared S-matrix:
    S = jnp.zeros((num_inputs + num_outputs, num_inputs + num_outputs), dtype=float)

    if not diagonal:
        S = S.at[:num_inputs, num_inputs:].set(1)
    else:
        r = jnp.arange(
            num_inputs,
            dtype=int,
        )  # == range(num_outputs) # reciprocal only works if num_inputs == num_outputs!
        S = S.at[r, num_inputs + r].set(1)

    if reciprocal:
        if not diagonal:
            S = S.at[num_inputs:, :num_inputs].set(1)
        else:
            # reciprocal only works if num_inputs == num_outputs!
            r = jnp.arange(num_inputs, dtype=int)  # == range(num_outputs)
            S = S.at[num_inputs + r, r].set(1)

    # let's convert it in SCOO format:
    Si, Sj = jnp.where(jnp.sqrt(sax.EPS) < S)
    Sx = S[Si, Sj]

    # the last missing piece is a port map:
    p = sax.PortNamer(num_inputs, num_outputs)
    pm = {
        **{p[i]: i for i in range(num_inputs)},
        **{p[i + num_inputs]: i + num_inputs for i in range(num_outputs)},
    }

    @validate_call
    def func(wl: sax.FloatArrayLike = 1.5) -> sax.SCoo:
        wl_ = jnp.asarray(wl)
        Sx_ = jnp.broadcast_to(Sx, (*wl_.shape, *Sx.shape))
        return Si, Sj, Sx_, pm

    func.__name__ = f"unitary_{num_inputs}_{num_outputs}"
    func.__qualname__ = f"unitary_{num_inputs}_{num_outputs}"
    return jax.jit(func)


@validate_call
def passthru(
    num_links: int,
    *,
    reciprocal: bool = True,
) -> sax.SCooModel:
    """Generate a multi-port pass-through device model.

    This function creates a model for devices that provide direct one-to-one
    connections between input and output ports with no coupling between
    different channels. Each input port connects directly to its corresponding
    output port, implementing a diagonal S-matrix.

    Pass-through devices are useful for modeling:
    - Fiber patch panels and interconnects
    - Optical switches in the "straight-through" state
    - Multi-channel delay lines or phase shifters
    - Ideal transmission links with no crosstalk

    Args:
        num_links: Number of independent pass-through links (input-output pairs).
            This creates a device with num_links inputs and num_links outputs.
        reciprocal: If True, the device exhibits reciprocal behavior where
            transmission is identical in both directions. This is typical for
            passive devices like fibers and waveguides. Defaults to True.

    Returns:
        A passthru model

    Examples:
        Create a 4-channel fiber ribbon cable:

        ```python
        import sax

        fiber_ribbon = sax.models.passthru(4, reciprocal=True)
        Si, Sj, Sx, port_map = fiber_ribbon(wl=1.55)
        # 4 independent channels with unity transmission
        ```

        Create an 8×8 optical switch (straight-through state):

        ```python
        switch_thru = sax.models.passthru(8, reciprocal=True)
        Si, Sj, Sx, port_map = switch_thru(wl=1.55)
        # Each input passes straight to corresponding output
        ```

        Create a unidirectional buffer array:

        ```python
        buffer_array = sax.models.passthru(6, reciprocal=False)
        Si, Sj, Sx, port_map = buffer_array(wl=1.55)
        # One-way transmission for each channel
        ```

        Multi-wavelength pass-through analysis:

        ```python
        import numpy as np

        wavelengths = np.linspace(1.5, 1.6, 101)
        multilink = sax.models.passthru(3, reciprocal=True)
        Si, Sj, Sx, port_map = multilink(wl=wavelengths)
        # Wavelength-independent transmission for all links
        ```

    Note:
        The pass-through model creates a diagonal unitary matrix where:
        - Each input connects only to its corresponding output
        - No crosstalk between different channels
        - Unity transmission (lossless) for each link
        - Perfect isolation between channels

        The S-matrix structure is:
        ```
        S = [0  I]  where I is the identity matrix
            [I* 0]  and I* is I if reciprocal else 0
        ```

        This model assumes:
        - Perfect channel isolation (infinite extinction ratio)
        - No wavelength dependence
        - No insertion loss
        - Ideal matching at all interfaces

        For realistic multi-channel devices, consider adding:
        - Channel-to-channel crosstalk
        - Wavelength-dependent transmission
        - Insertion loss and reflection
        - Polarization effects

        Applications include:
        - Modeling ideal optical fibers
        - Switch matrices in through state
        - Multi-channel delay lines
        - Parallel processing networks
        - Test and measurement setups
    """
    passthru = unitary(
        num_links,
        num_links,
        reciprocal=reciprocal,
        diagonal=True,
    )
    passthru.__name__ = f"passthru_{num_links}_{num_links}"
    passthru.__qualname__ = f"passthru_{num_links}_{num_links}"
    return jax.jit(passthru)
