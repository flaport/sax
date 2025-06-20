"""SAX Default Models."""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax


@cache
@validate_call
def model_2port(p1: sax.Name, p2: sax.Name) -> sax.SDictModel:
    """Generate a general 2-port model."""

    @jax.jit
    @validate_call
    def model_2port(wl: sax.FloatArrayLike = 1.5) -> sax.SDict:
        wl = jnp.asarray(wl)
        return sax.reciprocal({(p1, p2): jnp.ones_like(wl)})

    return model_2port


@cache
@validate_call
def model_3port(p1: sax.Name, p2: sax.Name, p3: sax.Name) -> sax.SDictModel:
    """Generate a general 3-port model."""

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


@cache
@validate_call
def model_4port(
    p1: sax.Name, p2: sax.Name, p3: sax.Name, p4: sax.Name
) -> sax.SDictModel:
    """Generate a general 4-port model."""

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


@cache
@validate_call
def unitary(
    num_inputs: int | None = None,
    num_outputs: int | None = None,
    ports: tuple[str, ...] | None = None,
    *,
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
    pm = {
        **{p: i for i, p in enumerate(input_ports)},
        **{p: i + num_inputs for i, p in enumerate(output_ports)},
    }

    @validate_call
    def func(*, wl: sax.FloatArrayLike = 1.5) -> sax.SCoo:
        wl_ = jnp.asarray(wl)
        Sx_ = jnp.broadcast_to(Sx, (*wl_.shape, *Sx.shape))
        return Si, Sj, Sx_, pm

    func.__name__ = f"unitary_{num_inputs}_{num_outputs}"
    func.__qualname__ = f"unitary_{num_inputs}_{num_outputs}"
    return jax.jit(func)


@cache
@validate_call
def copier(
    num_inputs: int | None = None,
    num_outputs: int | None = None,
    ports: tuple[str, ...] | None = None,
    *,
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
    Si, Sj = jnp.where(jnp.sqrt(sax.EPS) < S)
    Sx = S[Si, Sj]

    # the last missing piece is a port map:
    pm = {
        **{p: i for i, p in enumerate(input_ports)},
        **{p: i + num_inputs for i, p in enumerate(output_ports)},
    }

    @validate_call
    def func(wl: sax.FloatArrayLike = 1.5) -> sax.SCoo:
        wl_ = jnp.asarray(wl)
        Sx_ = jnp.broadcast_to(Sx, (*wl_.shape, *Sx.shape))
        return Si, Sj, Sx_, pm

    func.__name__ = f"unitary_{num_inputs}_{num_outputs}"
    func.__qualname__ = f"unitary_{num_inputs}_{num_outputs}"
    return jax.jit(func)


@cache
@validate_call
def passthru(
    num_links: int | None = None,
    ports: tuple[str, ...] | None = None,
    *,
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
        reciprocal=reciprocal,
        diagonal=True,
    )
    passthru.__name__ = f"passthru_{num_links}_{num_links}"
    passthru.__qualname__ = f"passthru_{num_links}_{num_links}"
    return jax.jit(passthru)


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
