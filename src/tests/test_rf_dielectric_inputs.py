"""Regression tests for real-only RF dielectric parameters."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sax.models import rf


@pytest.mark.parametrize("disable_jit", [True, False])
@pytest.mark.parametrize(
    "value",
    [
        11.9 - 0.119j,
        11.9 + 0j,
        np.complex128(11.9 - 0.119j),
        np.array([11.9, 11.9 - 0.119j]),
        jnp.array([11.9, 11.9 - 0.119j]),
    ],
    ids=["python", "zero-imaginary", "numpy-scalar", "numpy-array", "jax-array"],
)
@pytest.mark.parametrize(
    ("model", "parameter"),
    [
        (partial(rf.cpw_epsilon_eff, 20e-6, 15e-6, 100e-6), "ep_r"),
        (partial(rf.cpw_z0, 20e-6, 15e-6), "ep_eff"),
        (partial(rf.cpw_thickness_correction, 20e-6, 15e-6, 0.2e-6), "ep_eff"),
        (partial(rf.microstrip_epsilon_eff, 20e-6, 100e-6), "ep_r"),
        (partial(rf.microstrip_z0, 20e-6, 100e-6), "ep_eff"),
        (
            partial(
                rf.microstrip_thickness_correction, 20e-6, 100e-6, 0.2e-6, ep_eff=6.0
            ),
            "ep_r",
        ),
        (
            partial(
                rf.microstrip_thickness_correction, 20e-6, 100e-6, 0.2e-6, ep_r=11.9
            ),
            "ep_eff",
        ),
        (partial(rf.propagation_constant, 1e9), "ep_eff"),
        (partial(rf.propagation_constant, 1e9, 6.0), "ep_r"),
        (partial(rf.propagation_constant, 1e9, 6.0), "tand"),
        (rf.coplanar_waveguide, "ep_r"),
        (rf.coplanar_waveguide, "tand"),
        (rf.microstrip, "ep_r"),
        (rf.microstrip, "tand"),
    ],
)
def test_rejects_complex_dielectric_parameters(
    model: Callable,
    parameter: str,
    value: complex | np.ndarray | jax.Array,
    *,
    disable_jit: bool,
) -> None:
    with (
        jax.disable_jit(disable_jit),
        pytest.raises(TypeError, match=rf"{parameter} must be real; .*tand"),
    ):
        model(**{parameter: value})


@pytest.mark.parametrize("model", [rf.coplanar_waveguide, rf.microstrip])
@pytest.mark.parametrize("disable_jit", [True, False])
@pytest.mark.parametrize("tand", [0.01, np.array([0.001, 0.01, 0.03])])
def test_real_loss_tangent_attenuates(
    model: Callable, tand: float | np.ndarray, *, disable_jit: bool
) -> None:
    f = np.array([1e9, 10e9, 100e9])
    with jax.disable_jit(disable_jit):
        lossless = model(f=f, ep_r=11.9, tand=0.0)
        lossy = model(f=f, ep_r=11.9, tand=tand)
        scalar_results = [
            model(f=frequency, ep_r=11.9, tand=loss)["o1", "o2"]
            for frequency, loss in zip(f, np.broadcast_to(tand, f.shape), strict=True)
        ]

    assert_allclose(np.abs(lossless["o1", "o2"]), 1.0, atol=1e-12)
    assert np.all(np.abs(lossy["o1", "o2"]) < 1.0)
    assert_allclose(lossy["o1", "o2"], scalar_results, rtol=1e-12)
