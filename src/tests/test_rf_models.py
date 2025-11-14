import jax.numpy as jnp
import pytest

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

    def test_electrical_short(self, freq_array: jnp.ndarray) -> None:
        """Test electrical_short with frequency array."""
        s = rf.electrical_short(f=freq_array, n_ports=2)

        self._assert_s_params_dict(s, expected_shape=(len(freq_array),))
        self._assert_s_param(s, ("o1", "o1"), -1)
        self._assert_s_param(s, ("o2", "o2"), -1)
        self._assert_s_param(s, ("o1", "o2"), 0)

    def test_electrical_open(self, freq_array: jnp.ndarray) -> None:
        """Test electrical_open with frequency array."""
        s = rf.electrical_open(f=freq_array, n_ports=2)

        self._assert_s_params_dict(s, expected_shape=(len(freq_array),))
        self._assert_s_param(s, ("o1", "o1"), 1)
        self._assert_s_param(s, ("o2", "o2"), 1)
        self._assert_s_param(s, ("o1", "o2"), 0)
