"""SAX Crossing Models."""

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def crossing_ideal(wl: sax.FloatArrayLike = 1.5) -> sax.SDict:
    """Ideal waveguide crossing model.

    This function models an ideal 4-port waveguide crossing where light from each
    input port is transmitted to the corresponding output port on the opposite side
    with no loss, reflection, or crosstalk. This represents the theoretical limit
    of a perfect crossing design.

    The crossing has four ports arranged in a cross pattern:
    - Port o1 (left input) connects to Port o3 (right output)
    - Port o2 (bottom input) connects to Port o4 (top output)
    - No coupling between orthogonal waveguides (no crosstalk)
    - Perfect transmission with unit amplitude

    Args:
        wl: Operating wavelength in micrometers. Can be a scalar or array for
            multi-wavelength simulations. The wavelength dependence is minimal
            for this ideal model. Defaults to 1.5 μm.

    Returns:
        The crossing s-matrix

    Examples:
        Basic crossing simulation:

        ```python
        import sax

        # Single wavelength
        s_matrix = sax.models.crossing_ideal(wl=1.55)
        print(f"Horizontal transmission: {s_matrix[('o1', 'o3')]}")
        print(f"Vertical transmission: {s_matrix[('o2', 'o4')]}")
        ```

        Multi-wavelength analysis:

        ```python
        import numpy as np

        wavelengths = np.linspace(1.5, 1.6, 101)
        s_matrices = sax.models.crossing_ideal(wl=wavelengths)
        # All transmissions should be unity
        transmission = np.abs(s_matrices[("o1", "o3")]) ** 2
        ```

    Note:
        This is an idealized model that assumes:
        - Perfect phase matching between intersecting waveguides
        - No scattering losses at the intersection
        - No reflection at the crossing interfaces
        - Complete isolation between orthogonal paths

        Real crossings typically have finite insertion loss (0.1-1 dB),
        some crosstalk (-20 to -40 dB), and wavelength-dependent performance.
        For more realistic modeling, consider using fabrication-specific
        crossing models with measured parameters.
    """
    one = jnp.ones_like(jnp.asarray(wl))
    p = sax.PortNamer(2, 2)
    return sax.reciprocal(
        {
            (p.o1, p.o3): one,
            (p.o2, p.o4): one,
        }
    )
