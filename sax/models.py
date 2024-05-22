""" SAX Default Models """

from __future__ import annotations

from functools import lru_cache as cache
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .saxtypes import FloatArrayND, Model, SCoo, SDict
from .utils import get_inputs_outputs, reciprocal


def straight(
    *,
    wl: FloatArrayND | float = 1.55,
    wl0: float = 1.55,
    neff: float = 2.34,
    ng: float = 3.4,
    length: float = 10.0,
    loss: float = 0.0,
) -> SDict:
    """A simple straight waveguide model.

    Args:
        wl: wavelength in microns.
        wl0: reference wavelength in microns.
        neff: effective index.
        ng: group index.
        length: length of the waveguide in microns.
        loss: loss in dB/cm.

    """
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    _neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * _neff * length / wl
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    return reciprocal(
        {
            ("in0", "out0"): transmission,
        }
    )


def coupler(*, coupling: float = 0.5) -> SDict:
    """a simple coupler model"""
    kappa = coupling**0.5
    tau = (1 - coupling) ** 0.5
    return reciprocal(
        {
            ("in0", "out0"): tau,
            ("in0", "out1"): 1j * kappa,
            ("in1", "out0"): 1j * kappa,
            ("in1", "out1"): tau,
        }
    )


def _validate_ports(
    ports, num_inputs, num_outputs, diagonal
) -> Tuple[Tuple[str, ...], Tuple[str, ...], int, int]:
    """Validate the ports and return the input and output ports."""

    if ports is None:
        if num_inputs is None or num_outputs is None:
            raise ValueError(
                "if not ports given, you must specify how many input ports "
                "and how many output ports a model has."
            )
        input_ports = [f"in{i}" for i in range(num_inputs)]
        output_ports = [f"out{i}" for i in range(num_outputs)]
    else:
        if num_inputs is not None:
            if num_outputs is None:
                raise ValueError(
                    "if num_inputs is given, num_outputs should be given as well."
                )
        if num_outputs is not None:
            if num_inputs is None:
                raise ValueError(
                    "if num_outputs is given, num_inputs should be given as well."
                )
        if num_inputs is not None and num_outputs is not None:
            if num_inputs + num_outputs != len(ports):
                raise ValueError("num_inputs + num_outputs != len(ports)")
            input_ports = ports[:num_inputs]
            output_ports = ports[num_inputs:]
        else:
            input_ports, output_ports = get_inputs_outputs(ports)
            num_inputs = len(input_ports)
            num_outputs = len(output_ports)

    if diagonal:
        if num_inputs != num_outputs:
            raise ValueError(
                "Can only have a diagonal passthru if number "
                "of input ports equals the number of output ports!"
            )

    return tuple(input_ports), tuple(output_ports), num_inputs, num_outputs


@cache
def unitary(
    num_inputs: Optional[int] = None,
    num_outputs: Optional[int] = None,
    ports: Optional[Tuple[str, ...]] = None,
    *,
    jit=True,
    reciprocal=True,
    diagonal=False,
) -> Model:
    """A unitary model.

    Args:
        num_inputs: number of input ports.
        num_outputs: number of output ports.
        ports: tuple of input and output ports.
        jit: whether to jit the model.
        reciprocal: whether the model is reciprocal.
        diagonal: whether the model is diagonal.

    """
    input_ports, output_ports, num_inputs, num_outputs = _validate_ports(
        ports, num_inputs, num_outputs, diagonal
    )
    assert num_inputs is not None and num_outputs is not None

    # let's create the squared S-matrix:
    N = max(num_inputs, num_outputs)
    S = jnp.zeros((2 * N, 2 * N), dtype=float)

    if not diagonal:
        S = S.at[:N, N:].set(1)
    else:
        r = jnp.arange(
            N, dtype=int
        )  # reciprocal only works if num_inputs == num_outputs!
        S = S.at[r, N + r].set(1)

    if reciprocal:
        if not diagonal:
            S = S.at[N:, :N].set(1)
        else:
            r = jnp.arange(
                N, dtype=int
            )  # reciprocal only works if num_inputs == num_outputs!
            S = S.at[N + r, r].set(1)

    # Now we need to normalize the squared S-matrix
    U, s, V = jnp.linalg.svd(S, full_matrices=False)
    S = jnp.sqrt(U @ jnp.diag(jnp.where(s > 1e-12, 1, 0)) @ V)

    # Now create subset of this matrix we're interested in:
    r = jnp.concatenate(
        [jnp.arange(num_inputs, dtype=int), N + jnp.arange(num_outputs, dtype=int)], 0
    )
    S = S[r, :][:, r]

    # let's convert it in SCOO format:
    Si, Sj = jnp.where(S > 1e-6)
    Sx = S[Si, Sj]

    # the last missing piece is a port map:
    pm = {
        **{p: i for i, p in enumerate(input_ports)},
        **{p: i + num_inputs for i, p in enumerate(output_ports)},
    }

    def func(wl: float = 1.5) -> SCoo:
        wl_ = jnp.asarray(wl)
        Sx_ = jnp.broadcast_to(Sx, (*wl_.shape, *Sx.shape))
        return Si, Sj, Sx_, pm

    func.__name__ = f"unitary_{num_inputs}_{num_outputs}"
    func.__qualname__ = f"unitary_{num_inputs}_{num_outputs}"
    return jax.jit(func) if jit else func


@cache
def copier(
    num_inputs: Optional[int] = None,
    num_outputs: Optional[int] = None,
    ports: Optional[Tuple[str, ...]] = None,
    *,
    jit=True,
    reciprocal=True,
    diagonal=False,
) -> Model:
    """A copier model.

    Args:
        num_inputs: number of input ports.
        num_outputs: number of output ports.
        ports: tuple of input and output ports.
        jit: whether to jit the model.
        reciprocal: whether the model is reciprocal.
        diagonal: whether the model is diagonal.
    """
    input_ports, output_ports, num_inputs, num_outputs = _validate_ports(
        ports, num_inputs, num_outputs, diagonal
    )
    assert num_inputs is not None and num_outputs is not None

    # let's create the squared S-matrix:
    S = jnp.zeros((num_inputs + num_outputs, num_inputs + num_outputs), dtype=float)

    if not diagonal:
        S = S.at[:num_inputs, num_inputs:].set(1)
    else:
        r = jnp.arange(
            num_inputs, dtype=int
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
    Si, Sj = jnp.where(S > 1e-6)
    Sx = S[Si, Sj]

    # the last missing piece is a port map:
    pm = {
        **{p: i for i, p in enumerate(input_ports)},
        **{p: i + num_inputs for i, p in enumerate(output_ports)},
    }

    def func(wl: float = 1.5) -> SCoo:
        wl_ = jnp.asarray(wl)
        Sx_ = jnp.broadcast_to(Sx, (*wl_.shape, *Sx.shape))
        return Si, Sj, Sx_, pm

    func.__name__ = f"unitary_{num_inputs}_{num_outputs}"
    func.__qualname__ = f"unitary_{num_inputs}_{num_outputs}"
    return jax.jit(func) if jit else func


@cache
def passthru(
    num_links: Optional[int] = None,
    ports: Optional[Tuple[str, ...]] = None,
    *,
    jit=True,
    reciprocal=True,
) -> Model:
    """A passthru model.

    Args:
        num_links: number of links.
        ports: tuple of input and output ports.
        jit: whether to jit the model.
        reciprocal: whether the model is reciprocal.
    """
    passthru = unitary(
        num_links, num_links, ports, jit=jit, reciprocal=reciprocal, diagonal=True
    )
    passthru.__name__ = f"passthru_{num_links}_{num_links}"
    passthru.__qualname__ = f"passthru_{num_links}_{num_links}"
    return jax.jit(passthru) if jit else passthru


models = {
    "copier": copier,
    "coupler": coupler,
    "passthru": passthru,
    "straight": straight,
    "unitary": unitary,
}


def get_models(copy: bool = True):
    return {**models} if copy else models
