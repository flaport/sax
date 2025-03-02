"""SAX Default Models."""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax

from .constants import EPS
from .s import reciprocal

__all__ = ["copier", "coupler", "passthru", "straight", "unitary"]


@validate_call
def straight(
    *,
    wl: sax.FloatArrayLike = 1.55,
    wl0: sax.Float = 1.55,
    neff: sax.Float = 2.34,
    ng: sax.Float = 3.4,
    length: sax.Float = 10.0,
    loss: sax.Float = 0.0,
) -> sax.SDict:
    """A simple straight waveguide model.

    Args:
        wl: wavelength in microns.
        wl0: reference wavelength in microns.
        neff: effective index.
        ng: group index.
        length: length of the waveguide in microns.
        loss: loss in dB/cm.

    """
    dwl: sax.FloatArray = sax.into[sax.FloatArray](wl) - wl0
    dneff_dwl = (ng - neff) / wl0
    _neff = neff - dwl * dneff_dwl
    phase = 2 * jnp.pi * _neff * length / wl
    amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
    transmission = amplitude * jnp.exp(1j * phase)
    return reciprocal(
        {
            ("in0", "out0"): transmission,
        },
    )


@validate_call
def coupler(*, coupling: float = 0.5) -> sax.SDict:
    """A simple coupler model."""
    kappa = coupling**0.5
    tau = (1 - coupling) ** 0.5
    return reciprocal(
        {
            ("in0", "out0"): tau,
            ("in0", "out1"): 1j * kappa,
            ("in1", "out0"): 1j * kappa,
            ("in1", "out1"): tau,
        },
    )


@cache
@validate_call
def unitary(
    num_inputs: int | None = None,
    num_outputs: int | None = None,
    ports: tuple[str, ...] | None = None,
    *,
    jit: bool = True,
    reciprocal: bool = True,
    diagonal: bool = False,
) -> sax.SCooModel:
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
        ports,
        num_inputs,
        num_outputs,
        diagonal=diagonal,
    )

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
    S = jnp.sqrt(U @ jnp.diag(jnp.where(s > EPS, 1, 0)) @ V)

    # Now create subset of this matrix we're interested in:
    r = jnp.concatenate(
        [jnp.arange(num_inputs, dtype=int), N + jnp.arange(num_outputs, dtype=int)],
        0,
    )
    S = S[r, :][:, r]

    # let's convert it in SCOO format:
    Si, Sj = jnp.where(S > EPS)
    Sx = S[Si, Sj]

    # the last missing piece is a port map:
    pm = {
        **{p: i for i, p in enumerate(input_ports)},
        **{p: i + num_inputs for i, p in enumerate(output_ports)},
    }

    def func(*, wl: sax.Float = 1.5) -> sax.SCoo:
        wl_ = jnp.asarray(wl)
        Sx_ = jnp.broadcast_to(Sx, (*wl_.shape, *Sx.shape))
        return Si, Sj, Sx_, pm

    func.__name__ = f"unitary_{num_inputs}_{num_outputs}"
    func.__qualname__ = f"unitary_{num_inputs}_{num_outputs}"
    return jax.jit(func) if jit else func


@cache
@validate_call
def copier(
    num_inputs: int | None = None,
    num_outputs: int | None = None,
    ports: tuple[str, ...] | None = None,
    *,
    jit: bool = True,
    reciprocal: bool = True,
    diagonal: bool = False,
) -> sax.SCooModel:
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
        ports,
        num_inputs,
        num_outputs,
        diagonal=diagonal,
    )

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
    Si, Sj = jnp.where(S > jnp.sqrt(EPS))
    Sx = S[Si, Sj]

    # the last missing piece is a port map:
    pm = {
        **{p: i for i, p in enumerate(input_ports)},
        **{p: i + num_inputs for i, p in enumerate(output_ports)},
    }

    def func(wl: float = 1.5) -> sax.SCoo:
        wl_ = jnp.asarray(wl)
        Sx_ = jnp.broadcast_to(Sx, (*wl_.shape, *Sx.shape))
        return Si, Sj, Sx_, pm

    func.__name__ = f"unitary_{num_inputs}_{num_outputs}"
    func.__qualname__ = f"unitary_{num_inputs}_{num_outputs}"
    return jax.jit(func) if jit else func


@cache
@validate_call
def passthru(
    num_links: int | None = None,
    ports: tuple[str, ...] | None = None,
    *,
    jit: bool = True,
    reciprocal: bool = True,
) -> sax.SCooModel:
    """A passthru model.

    Args:
        num_links: number of links.
        ports: tuple of input and output ports.
        jit: whether to jit the model.
        reciprocal: whether the model is reciprocal.
    """
    passthru = unitary(
        num_links,
        num_links,
        ports,
        jit=jit,
        reciprocal=reciprocal,
        diagonal=True,
    )
    passthru.__name__ = f"passthru_{num_links}_{num_links}"
    passthru.__qualname__ = f"passthru_{num_links}_{num_links}"
    return jax.jit(passthru) if jit else passthru


def _validate_ports(
    ports: tuple[sax.Port, ...] | None,
    num_inputs: int | None,
    num_outputs: int | None,
    *,
    diagonal: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], int, int]:
    """Validate the ports and return the input and output ports."""
    if ports is None:
        if num_inputs is None or num_outputs is None:
            msg = (
                "if not ports given, you must specify how many input ports "
                "and how many output ports a model has."
            )
            raise ValueError(
                msg,
            )
        input_ports = [f"in{i}" for i in range(num_inputs)]
        output_ports = [f"out{i}" for i in range(num_outputs)]
    else:
        if num_inputs is not None and num_outputs is None:
            msg = "if num_inputs is given, num_outputs should be given as well."
            raise ValueError(
                msg,
            )
        if num_outputs is not None and num_inputs is None:
            msg = "if num_outputs is given, num_inputs should be given as well."
            raise ValueError(
                msg,
            )
        if num_inputs is not None and num_outputs is not None:
            if num_inputs + num_outputs != len(ports):
                msg = "num_inputs + num_outputs != len(ports)"
                raise ValueError(msg)
            input_ports = ports[:num_inputs]
            output_ports = ports[num_inputs:]
        else:
            input_ports, output_ports = _get_inputs_outputs(ports)
            num_inputs = len(input_ports)
            num_outputs = len(output_ports)

    if diagonal and num_inputs != num_outputs:
        msg = (
            "Can only have a diagonal passthru if number "
            "of input ports equals the number of output ports!"
        )
        raise ValueError(
            msg,
        )

    return tuple(input_ports), tuple(output_ports), num_inputs, num_outputs


def _get_inputs_outputs(
    ports: tuple[sax.Port, ...],
) -> tuple[tuple[sax.Port, ...], tuple[sax.Port, ...]]:
    inputs = tuple(p for p in ports if p.lower().startswith("in"))
    outputs = tuple(p for p in ports if not p.lower().startswith("in"))
    if not inputs:
        inputs = tuple(p for p in ports if not p.lower().startswith("out"))
        outputs = tuple(p for p in ports if p.lower().startswith("out"))
    return inputs, outputs
