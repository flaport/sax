from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import pytest
from jax.typing import ArrayLike

import sax
from sax.models import rf


class TestRFModels:
    """Test suite for RF model components."""

    @pytest.fixture
    def freq_array(self) -> jnp.ndarray:
        """Frequency array fixture."""
        return jnp.linspace(1e9, 10e9, 10)

    @pytest.fixture
    def freq_single(self) -> float:
        """Single frequency fixture."""
        return 1e9

    @staticmethod
    def _assert_s_params_dict(s: dict, expected_shape: tuple | None = None) -> None:
        """Helper method to assert common S-parameter dictionary properties."""
        assert isinstance(s, dict)
        if expected_shape is not None:
            assert s[("o1", "o1")].shape == expected_shape

    @staticmethod
    def _assert_s_param(s: dict, port_pair: tuple, expected_value: complex) -> None:
        """Helper method to assert a specific S-parameter value."""
        assert jnp.allclose(s[port_pair], expected_value)

    def test_gamma_0_load(self, freq_array: jnp.ndarray) -> None:
        """Test gamma_0_load with frequency array."""
        s = rf.gamma_0_load(f=freq_array, gamma_0=0.5, n_ports=2)

        self._assert_s_params_dict(s, expected_shape=(len(freq_array),))
        self._assert_s_param(s, ("o1", "o1"), 0.5)
        self._assert_s_param(s, ("o1", "o2"), 0)

    def test_tee(self, freq_array: jnp.ndarray) -> None:
        """Test tee splitter."""
        s = rf.tee(f=freq_array)

        self._assert_s_params_dict(s, expected_shape=(len(freq_array),))
        self._assert_s_param(s, ("o1", "o1"), -1 / 3)
        self._assert_s_param(s, ("o1", "o2"), 2 / 3)

    def test_impedance(self, freq_single: float) -> None:
        """Test impedance element."""
        s = rf.impedance(f=freq_single, z=75, z0=50)

        self._assert_s_params_dict(s, expected_shape=())
        self._assert_s_param(s, ("o1", "o1"), 75 / (75 + 100))

    def test_admittance(self, freq_single: float) -> None:
        """Test admittance element."""
        s = rf.admittance(f=freq_single, y=1 / 75)

        self._assert_s_params_dict(s, expected_shape=())
        self._assert_s_param(s, ("o1", "o1"), 1 / (1 + 1 / 75))

    def test_capacitor(self, freq_single: float) -> None:
        """Test capacitor element."""
        s = rf.capacitor(f=freq_single, capacitance=1e-12, z0=50)

        self._assert_s_params_dict(s, expected_shape=())

        angular_frequency = 2 * jnp.pi * freq_single
        capacitor_impedance = 1 / (1j * angular_frequency * 1e-12)
        expected_s11 = capacitor_impedance / (capacitor_impedance + 100)
        self._assert_s_param(s, ("o1", "o1"), expected_s11)

    def test_inductor(self, freq_single: float) -> None:
        """Test inductor element."""
        s = rf.inductor(f=freq_single, inductance=1e-9, z0=50)

        self._assert_s_params_dict(s)

        angular_frequency = 2 * jnp.pi * freq_single
        inductor_impedance = 1j * angular_frequency * 1e-9
        expected_s11 = inductor_impedance / (inductor_impedance + 100)
        self._assert_s_param(s, ("o1", "o1"), expected_s11)

    @pytest.mark.parametrize("n_ports", [1, 2, 3])
    def test_electrical_short(self, freq_array: jnp.ndarray, n_ports: int) -> None:
        """Test electrical_short with frequency array."""
        s = rf.electrical_short(f=freq_array, n_ports=n_ports)

        self._assert_s_params_dict(s, expected_shape=(len(freq_array),))
        for i, j in product(range(1, n_ports + 1), repeat=2):
            expected_value = -1 if i == j else 0
            self._assert_s_param(s, (f"o{i}", f"o{j}"), expected_value)

    @pytest.mark.parametrize("n_ports", [1, 2, 3])
    def test_electrical_open(self, freq_array: jnp.ndarray, n_ports: int) -> None:
        """Test electrical_open with frequency array."""
        s = rf.electrical_open(f=freq_array, n_ports=n_ports)

        self._assert_s_params_dict(s, expected_shape=(len(freq_array),))
        for i, j in product(range(1, n_ports + 1), repeat=2):
            expected_value = 1 if i == j else 0
            self._assert_s_param(s, (f"o{i}", f"o{j}"), expected_value)

    @staticmethod
    @partial(jax.jit, static_argnames=("L", "C", "Z0"))
    def lc_shunt_component(
        f: ArrayLike = jnp.array([5e9]),  # noqa: B008
        L: sax.FloatLike = 1e-9,
        C: sax.FloatLike = 1e-12,
        Z0: sax.FloatLike = 50,
    ) -> sax.SDict:
        """SAX component for a 1-port shunted LC resonator."""
        f = jnp.asarray(f)
        models = {
            "L": rf.inductor,
            "C": rf.capacitor,
            "short": rf.electrical_short,
            "tee": rf.tee,
        }

        circuit, _ = sax.circuit(
            netlist={
                "instances": {
                    "L": {
                        "component": "L",
                        "settings": {"inductance": L, "z0": Z0},
                    },
                    "C": {
                        "component": "C",
                        "settings": {"capacitance": C, "z0": Z0},
                    },
                    "tee_1": {
                        "component": "tee",
                    },
                    "tee_2": {
                        "component": "tee",
                    },
                    "gnd": {
                        "component": "short",
                    },
                },
                "connections": {
                    "L,o1": "tee_1,o1",
                    "C,o1": "tee_1,o2",
                    "L,o2": "tee_2,o1",
                    "C,o2": "tee_2,o2",
                    "gnd,o1": "tee_2,o3",
                },
                "ports": {
                    "o1": "tee_1,o3",
                },
            },
            models=models,
        )

        return circuit(f=f)

    def test_lc_shunt_component(self, freq_array: jnp.ndarray) -> None:
        """Test LC shunt component circuit."""
        s = type(self).lc_shunt_component(
            f=freq_array,
            L=1e-9,
            C=1e-12,
            Z0=50,
        )

        self._assert_s_params_dict(s, expected_shape=(len(freq_array),))
