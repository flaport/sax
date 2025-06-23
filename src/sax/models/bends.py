"""SAX Bend Models."""

import jax
from pydantic import validate_call

import sax

from .straight import straight


@jax.jit
@validate_call
def bend(
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.FloatArrayLike = 1.55,
    neff: sax.FloatArrayLike = 2.34,
    ng: sax.FloatArrayLike = 3.4,
    length: sax.FloatArrayLike = 10.0,
    loss_dB_cm: sax.FloatArrayLike = 0.1,
) -> sax.SDict:
    """Simple waveguide bend model.

    This function models a simple bend in a waveguide by treating it as an
    equivalent straight waveguide with the same effective length and properties.
    The bend is approximated using the straight waveguide model, inheriting all
    its dispersion and loss characteristics.

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
        The bend s-matrix

    Examples:
        Basic bend simulation:

    ```python
    import numpy as np
    import sax

    # Single wavelength simulation
    s_matrix = sax.models.bend(wl=1.55, length=20.0, loss_dB_cm=0.2)
    print(f"Transmission: {s_matrix[('in0', 'out0')]}")
    ```

    Multi-wavelength analysis:

    ```python
    wavelengths = np.linspace(1.5, 1.6, 101)
    s_matrices = sax.models.bend(
        wl=wavelengths, length=50.0, loss_dB_cm=0.1, neff=2.35, ng=3.5
    )
    transmission = np.abs(s_matrices[("in0", "out0")]) ** 2
    ```

    ```
                   o2/out0
                   |
                  /
                 /
    o1/in0 _____/
    ```

    ```python
    # mkdocs: render
    import matplotlib.pyplot as plt
    import numpy as np
    import sax

    sax.set_port_naming_strategy("optical")

    wavelengths = np.linspace(1.5, 1.6, 101)
    s = sax.models.bend(wl=wavelengths, length=50.0, loss_dB_cm=0.1, neff=2.35, ng=3.5)
    transmission = np.abs(s[("o1", "o2")]) ** 2

    plt.figure()
    plt.plot(wavelengths, transmission)
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Transmission")
    ```

    Note:
        This model treats the bend as an equivalent straight waveguide and does not
        account for:
        - Mode coupling between bend eigenmodes
        - Radiation losses due to bending
        - Polarization effects in bends
        - Non-linear dispersion effects

        For more accurate bend modeling, consider using dedicated bend models that
        account for these physical effects.
    """
    return straight(
        wl=wl,
        wl0=wl0,
        neff=neff,
        ng=ng,
        length=length,
        loss_dB_cm=loss_dB_cm,
    )
