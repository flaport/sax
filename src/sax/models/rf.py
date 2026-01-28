"""Sax generic RF models."""

from functools import partial

import jax
import jax.numpy as jnp

import sax


@partial(jax.jit, static_argnames=("n_ports",))
def gamma_0_load(
    f: sax.FloatArrayLike = 5e9,
    gamma_0: sax.ComplexLike = 0,
    n_ports: sax.IntLike = 1,
) -> sax.SDict:
    r"""Connection with given reflection coefficient.

    Args:
        f: Array of frequency points in Hz
        gamma_0: Reflection coefficient Γ₀ of connection
        n_ports: Number of ports in component. The diagonal ports of the matrix
            are set to Γ₀ and the off-diagonal ports to 0.

    Returns:
        S-dictionary where :math:`S = \Gamma_0I_\text{n\_ports}`

    Examples:
        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        f = np.linspace(1e9, 10e9, 500)
        gamma_0 = 0.5 * np.exp(1j * np.pi / 4)
        s = sax.models.rf.gamma_0_load(f=f, gamma_0=gamma_0, n_ports=2)
        plt.figure()
        plt.plot(f / 1e9, np.abs(s[("o1", "o1")]), label="|S11|")
        plt.plot(f / 1e9, np.abs(s[("o2", "o2")]), label="|S22|")
        plt.plot(f / 1e9, np.abs(s[("o1", "o2")]), label="|S12|")
        plt.plot(f / 1e9, np.abs(s[("o2", "o1")]), label="|S21|")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Magnitude")
        plt.legend()
        ```
    """
    f = jnp.asarray(f)
    sdict = {
        (f"o{i}", f"o{i}"): jnp.full(len(f), gamma_0) for i in range(1, n_ports + 1)
    }
    sdict |= {
        (f"o{i}", f"o{j}"): jnp.zeros(len(f), dtype=complex)
        for i in range(1, n_ports + 1)
        for j in range(i + 1, n_ports + 1)
    }
    return sax.reciprocal(sdict)


@jax.jit
def tee(f: sax.FloatArrayLike = 5e9) -> sax.SDict:
    """Ideal three-port RF power divider/combiner (T-junction).

    Args:
        f: Array of frequency points in Hz

    Returns:
        S-dictionary representing ideal RF T-junction behavior

    Examples:
        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        f = np.linspace(1e9, 10e9, 500)
        s = sax.models.rf.tee(f=f)
        plt.figure()
        plt.plot(f / 1e9, np.abs(s[("o1", "o2")]) ** 2, label="|S12|^2")
        plt.plot(f / 1e9, np.abs(s[("o1", "o3")]) ** 2, label="|S13|^2")
        plt.plot(f / 1e9, np.abs(s[("o2", "o3")]) ** 2, label="|S23|^2")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Power")
        plt.legend()
        ```
    """
    f = jnp.asarray(f)
    sdict = {(f"o{i}", f"o{i}"): jnp.full(len(f), -1 / 3) for i in range(1, 4)}
    sdict |= {
        (f"o{i}", f"o{j}"): jnp.full(len(f), 2 / 3)
        for i in range(1, 4)
        for j in range(i + 1, 4)
    }
    return sax.reciprocal(sdict)


@jax.jit
def impedance(
    f: sax.FloatArrayLike, z: sax.ComplexLike = 50, z0: sax.ComplexLike = 50
) -> sax.SDict:
    r"""Generalized two-port impedance element.

    Args:
        f: Frequency in Hz
        z: Impedance in Ω
        z0: Reference impedance in Ω.

    Returns:
        S-dictionary representing the impedance element

    References:
        [@pozar2012]

    Examples:
        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        f = np.linspace(1e9, 10e9, 500)
        s = sax.models.rf.impedance(f=f, z=75, z0=50)
        plt.figure()
        plt.plot(f / 1e9, np.abs(s[("o1", "o1")]), label="|S11|")
        plt.plot(f / 1e9, np.abs(s[("o1", "o2")]), label="|S12|")
        plt.plot(f / 1e9, np.abs(s[("o2", "o2")]), label="|S22|")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Magnitude")
        plt.legend()
        ```
    """
    f = jnp.asarray(f)
    sdict = {
        ("o1", "o1"): jnp.full(f.shape, z / (z + 2 * z0)),
        ("o1", "o2"): jnp.full(f.shape, 2 * z0 / (2 * z0 + z)),
        ("o2", "o2"): jnp.full(f.shape, z / (z + 2 * z0)),
    }
    return sax.reciprocal(sdict)


@jax.jit
def admittance(f: sax.FloatArrayLike, y: sax.ComplexLike = 1 / 50) -> sax.SDict:
    r"""Generalized two-port admittance element.

    Args:
        f: Frequency in Hz
        y: Admittance in siemens

    Returns:
        S-dictionary representing the admittance element

    References:
        [@pozar2012]

    Examples:
        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        f = np.linspace(1e9, 10e9, 500)
        s = sax.models.rf.admittance(f=f, y=1 / 75)
        plt.figure()
        plt.plot(f / 1e9, np.abs(s[("o1", "o1")]), label="|S11|")
        plt.plot(f / 1e9, np.abs(s[("o1", "o2")]), label="|S12|")
        plt.plot(f / 1e9, np.abs(s[("o2", "o2")]), label="|S22|")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Magnitude")
        plt.legend()
        ```
    """
    f = jnp.asarray(f)
    sdict = {
        ("o1", "o1"): jnp.full(f.shape, 1 / (1 + y)),
        ("o1", "o2"): jnp.full(f.shape, y / (1 + y)),
        ("o2", "o2"): jnp.full(f.shape, 1 / (1 + y)),
    }
    return sax.reciprocal(sdict)


@partial(jax.jit, static_argnames=("capacitance",))
def capacitor(
    f: sax.FloatArrayLike = 5e9,
    capacitance: sax.FloatLike = 1e-15,
    z0: sax.ComplexLike = 50,
) -> sax.SDict:
    r"""Ideal two-port capacitor model.

    Args:
        f: Frequency in Hz
        capacitance: Capacitance in Farads
        z0: Reference impedance in Ω.

    Returns:
        S-dictionary representing the capacitor element

    References:
        [@pozar2012]

    Examples:
        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        f = np.linspace(1e9, 10e9, 500)
        s = sax.models.rf.capacitor(f=f, capacitance=1e-12, z0=50)
        plt.figure()
        plt.plot(f / 1e9, np.abs(s[("o1", "o1")]), label="|S11|")
        plt.plot(f / 1e9, np.abs(s[("o1", "o2")]), label="|S12|")
        plt.plot(f / 1e9, np.abs(s[("o2", "o2")]), label="|S22|")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Magnitude")
        plt.legend()
        ```
    """
    angular_frequency = 2 * jnp.pi * jnp.asarray(f)
    capacitor_impedance = 1 / (1j * angular_frequency * capacitance)
    return impedance(f=f, z=capacitor_impedance, z0=z0)


@partial(jax.jit, static_argnames=("inductance",))
def inductor(
    f: sax.FloatArrayLike = 5e9,
    inductance: sax.FloatLike = 1e-12,
    z0: sax.ComplexLike = 50,
) -> sax.SDict:
    r"""Ideal two-port inductor model.

    Args:
        f: Frequency in Hz
        inductance: Inductance in Henries
        z0: Reference impedance in Ω.

    Returns:
        S-dictionary representing the inductor element

    References:
        [@pozar2012]

    Examples:
        ```python
        # mkdocs: render
        import matplotlib.pyplot as plt
        import numpy as np
        import sax

        sax.set_port_naming_strategy("optical")

        f = np.linspace(1e9, 10e9, 500)
        s = sax.models.rf.inductor(f=f, inductance=1e-9, z0=50)
        plt.figure()
        plt.plot(f / 1e9, np.abs(s[("o1", "o1")]), label="|S11|")
        plt.plot(f / 1e9, np.abs(s[("o1", "o2")]), label="|S12|")
        plt.plot(f / 1e9, np.abs(s[("o2", "o2")]), label="|S22|")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Magnitude")
        plt.legend()
        ```
    """
    angular_frequency = 2 * jnp.pi * jnp.asarray(f)
    inductor_impedance = 1j * angular_frequency * inductance
    return impedance(f=f, z=inductor_impedance, z0=z0)


@partial(jax.jit, inline=True, static_argnames=("n_ports"))
def electrical_short(
    f: sax.FloatArrayLike = 5e9,
    n_ports: int = 1,
) -> sax.SDict:
    r"""Electrical short connection Sax model.

    Args:
        f: Array of frequency points in Hz
        n_ports: Number of ports to set as shorted

    Returns:
        S-dictionary where :math:`S = -I_\text{n\_ports}`

    References:
        [@pozar2012]
    """
    return gamma_0_load(f=f, gamma_0=-1, n_ports=n_ports)  # type: ignore[return-value]


@partial(jax.jit, inline=True, static_argnames=("n_ports"))
def electrical_open(
    f: sax.FloatArrayLike = 5e9,
    n_ports: int = 1,
) -> sax.SDict:
    r"""Electrical open connection Sax model.

    Useful for specifying some ports to remain open while not exposing
    them for connections in circuits.

    Args:
        f: Array of frequency points in Hz
        n_ports: Number of ports to set as opened

    Returns:
        S-dictionary where :math:`S = I_\text{n\_ports}`

    References:
        [@pozar2012]
    """
    return gamma_0_load(f=f, gamma_0=1, n_ports=n_ports)  # type: ignore[return-value]
