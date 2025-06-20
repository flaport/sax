"""SAX Default Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@jax.jit
@validate_call
def splitter_ideal(*, coupling: sax.FloatArrayLike = 0.5) -> sax.SDict:
    """Ideal 1x2 power splitter model.

    This function models an ideal 1x2 power splitter that divides input optical
    power between two output ports according to a specified coupling ratio.
    The model assumes no insertion loss and perfect reciprocity.

    The splitter has three ports:
    - One input port (in0)
    - Two output ports (out0, out1)

    Power splitting is determined by the coupling parameter:
    - Fraction (1-coupling) goes to out0
    - Fraction coupling goes to out1
    - Total power is conserved (no loss)

    Args:
        coupling: Power coupling ratio between 0 and 1. Determines the fraction
            of input power that couples to the second output port (out1).
            - coupling=0.0: All power to out0 (no splitting)
            - coupling=0.5: Equal power splitting (3dB splitter)
            - coupling=1.0: All power to out1
            Defaults to 0.5 for equal splitting.

    Returns:
        S-matrix dictionary containing the complex-valued cross/thru coefficients.

    Examples:
        Equal power splitter (3dB splitter):

        ```python
        import sax

        # 50/50 power splitter
        s_matrix = sax.models.splitter_ideal(coupling=0.5)
        print(f"Power to out0: {abs(s_matrix[('in0', 'out0')]) ** 2}")
        print(f"Power to out1: {abs(s_matrix[('in0', 'out1')]) ** 2}")
        # Both should be 0.5
        ```

        Asymmetric splitter:

        ```python
        # 90/10 power splitter
        s_matrix = sax.models.splitter_ideal(coupling=0.1)
        power_out0 = abs(s_matrix[("in0", "out0")]) ** 2  # Should be 0.9
        power_out1 = abs(s_matrix[("in0", "out1")]) ** 2  # Should be 0.1
        ```

        Variable coupling analysis:

        ```python
        import numpy as np

        couplings = np.linspace(0, 1, 101)
        power_ratios = []
        for c in couplings:
            s = sax.models.splitter_ideal(coupling=c)
            ratio = abs(s[("in0", "out1")]) ** 2 / abs(s[("in0", "out0")]) ** 2
            power_ratios.append(ratio)
        ```

    Note:
        This is an idealized lossless model that assumes:
        - Perfect power conservation (no insertion loss)
        - No reflection at any port
        - Wavelength-independent behavior
        - Perfect reciprocity

        Real splitters typically have some insertion loss (0.1-0.5 dB),
        wavelength-dependent splitting ratios, and may exhibit some reflection.
        The model uses amplitude coefficients (square root of power ratios)
        to ensure proper S-matrix properties.
    """
    kappa = jnp.asarray(coupling**0.5)
    tau = jnp.asarray((1 - coupling) ** 0.5)

    p = sax.PortNamer(num_inputs=1, num_outputs=2)
    return sax.reciprocal(
        {
            (p.in0, p.out0): tau,
            (p.in0, p.out1): kappa,
        },
    )
