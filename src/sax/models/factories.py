"""SAX Default Models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from pydantic import validate_call

import sax

from .ports import PortNamer


@validate_call
def model_2port(p1: sax.Name, p2: sax.Name) -> sax.SDictModel:
    """Generate a general 2-port model."""

    @jax.jit
    @validate_call
    def model_2port(wl: sax.FloatArrayLike = 1.5) -> sax.SDict:
        wl = jnp.asarray(wl)
        return sax.reciprocal({(p1, p2): jnp.ones_like(wl)})

    return model_2port


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


@validate_call
def unitary(
    num_inputs: int,
    num_outputs: int,
    *,
    reciprocal: bool = True,
    diagonal: bool = False,
) -> sax.SCooModel:
    """A unitary model."""
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
    p = PortNamer(num_inputs, num_outputs)
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
    """A copier model."""
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
    p = PortNamer(num_inputs, num_outputs)
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
    """A passthru model."""
    passthru = unitary(
        num_links,
        num_links,
        reciprocal=reciprocal,
        diagonal=True,
    )
    passthru.__name__ = f"passthru_{num_links}_{num_links}"
    passthru.__qualname__ = f"passthru_{num_links}_{num_links}"
    return jax.jit(passthru)
