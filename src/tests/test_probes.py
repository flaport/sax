"""Tests for the probes feature."""

import pytest

import sax


def test_ideal_probe_model() -> None:
    """Test that the ideal_probe model returns the expected S-matrix."""
    s = sax.models.ideal_probe(wl=1.55)

    # Check all expected ports exist
    expected_ports = {"in", "out", "tap_fwd", "tap_bwd"}
    ports = sax.get_ports(s)
    assert set(ports) == expected_ports

    # Check through path has full transmission
    assert float(s[("in", "out")]) == 1.0
    assert float(s[("out", "in")]) == 1.0

    # Check tap coupling
    assert float(s[("in", "tap_fwd")]) == 1.0
    assert float(s[("tap_fwd", "in")]) == 1.0
    assert float(s[("out", "tap_bwd")]) == 1.0
    assert float(s[("tap_bwd", "out")]) == 1.0

    # Check no cross-coupling between taps
    assert float(s[("tap_fwd", "tap_bwd")]) == 0.0
    assert float(s[("tap_bwd", "tap_fwd")]) == 0.0

    # Check no reflections
    assert float(s[("in", "in")]) == 0.0
    assert float(s[("out", "out")]) == 0.0
    assert float(s[("tap_fwd", "tap_fwd")]) == 0.0
    assert float(s[("tap_bwd", "tap_bwd")]) == 0.0


def test_basic_probe_insertion() -> None:
    """Test basic probe insertion on a simple 2-waveguide circuit."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    # Without probes
    circuit_no_probe, _ = sax.circuit(netlist, models)
    result_no_probe = circuit_no_probe()
    ports_no_probe = sax.get_ports(result_no_probe)
    assert set(ports_no_probe) == {"in", "out"}

    # With probe
    circuit_with_probe, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg1,out0"},
    )
    result_with_probe = circuit_with_probe()
    ports_with_probe = sax.get_ports(result_with_probe)
    assert set(ports_with_probe) == {"in", "out", "mid_fwd", "mid_bwd"}


def test_probe_on_right_side_of_connection() -> None:
    """Test that probes work when specified on the right side of a connection."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    # Probe on right side of connection (wg2,in0 instead of wg1,out0)
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg2,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert set(ports) == {"in", "out", "mid_fwd", "mid_bwd"}


def test_probe_fwd_direction_left_side() -> None:
    """Test _fwd captures signal flowing INTO user-specified port (left side)."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    # Probe at wg1,out0 (left side of connection)
    # _fwd should capture signal flowing INTO wg1,out0 (i.e., from wg2 back to wg1)
    # _bwd should capture signal flowing OUT of wg1,out0 (i.e., from wg1 to wg2)
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg1,out0"},
    )
    result = circuit_fn()

    # Signal from "in" goes through wg1 and exits at wg1,out0
    # This should appear at mid_bwd (signal flowing OUT of the probed port)
    assert ("in", "mid_bwd") in result

    # Signal from "out" goes back through wg2 and into wg1,out0
    # This should appear at mid_fwd (signal flowing INTO the probed port)
    assert ("out", "mid_fwd") in result


def test_probe_fwd_direction_right_side() -> None:
    """Test _fwd captures signal flowing INTO user-specified port (right side)."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    # Probe at wg2,in0 (right side of connection)
    # _fwd should capture signal flowing INTO wg2,in0 (i.e., from wg1 to wg2)
    # _bwd should capture signal flowing OUT of wg2,in0 (i.e., from wg2 back to wg1)
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg2,in0"},
    )
    result = circuit_fn()

    # Signal from "in" goes through wg1 and into wg2,in0
    # This should appear at mid_fwd (signal flowing INTO the probed port)
    assert ("in", "mid_fwd") in result

    # Signal from "out" goes back through wg2 and exits at wg2,in0
    # This should appear at mid_bwd (signal flowing OUT of the probed port)
    assert ("out", "mid_bwd") in result


def test_multiple_probes() -> None:
    """Test multiple probes on different connections."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
            "wg3": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
            "wg2,out0": "wg3,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg3,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={
            "probe1": "wg1,out0",
            "probe2": "wg2,out0",
        },
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    expected_ports = {
        "in",
        "out",
        "probe1_fwd",
        "probe1_bwd",
        "probe2_fwd",
        "probe2_bwd",
    }
    assert set(ports) == expected_ports


def test_probe_in_mzi_circuit() -> None:
    """Test probe in a more complex MZI circuit."""
    netlist = {
        "instances": {
            "lft": "coupler",
            "top": "waveguide",
            "btm": "waveguide",
            "rgt": "coupler",
        },
        "connections": {
            "lft,out0": "btm,in0",
            "btm,out0": "rgt,in0",
            "lft,out1": "top,in0",
            "top,out0": "rgt,in1",
        },
        "ports": {
            "in0": "lft,in0",
            "in1": "lft,in1",
            "out0": "rgt,out0",
            "out1": "rgt,out1",
        },
    }

    models = {
        "coupler": sax.models.coupler_ideal,
        "waveguide": sax.models.straight,
    }

    # Probe on top arm
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"top_arm": "top,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    expected_ports = {"in0", "in1", "out0", "out1", "top_arm_fwd", "top_arm_bwd"}
    assert set(ports) == expected_ports


def test_probe_values_match_transmission() -> None:
    """Test that probe values match expected transmission through the circuit."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg1,out0"},
    )

    # Evaluate at a specific wavelength
    result = circuit_fn(wl=1.55)

    # The forward probe should capture signal going from wg1 to wg2
    # The transmission in->out should still work
    assert ("in", "out") in result
    assert ("in", "mid_fwd") in result
    assert ("out", "mid_bwd") in result


def test_probe_error_on_non_connected_port() -> None:
    """Test that probes raise error when referencing non-connected ports."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
        },
        "connections": {},
        "ports": {
            "in": "wg1,in0",
            "out": "wg1,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    with pytest.raises(ValueError, match="not part of any connection"):
        sax.circuit(
            netlist,
            models,
            probes={"mid": "wg1,out0"},
        )


def test_probe_error_on_port_conflict() -> None:
    """Test that probes raise error when port names would conflict."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
            "mid_fwd": "wg2,out0",  # Conflict with probe port
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    with pytest.raises(ValueError, match="conflict with existing ports"):
        sax.circuit(
            netlist,
            models,
            probes={"mid": "wg1,out0"},
        )


def test_probe_error_on_instance_conflict() -> None:
    """Test that probes raise error when instance name would conflict."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
            "_probe_mid": "waveguide",  # Conflict with probe instance name
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    with pytest.raises(ValueError, match="conflicts with an existing instance"):
        sax.circuit(
            netlist,
            models,
            probes={"mid": "wg1,out0"},
        )


def test_empty_probes_dict() -> None:
    """Test that empty probes dict is a no-op."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    circuit_fn, _ = sax.circuit(netlist, models, probes={})
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert set(ports) == {"in", "out"}


def test_probe_with_none() -> None:
    """Test that probes=None is a no-op."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    circuit_fn, _ = sax.circuit(netlist, models, probes=None)
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert set(ports) == {"in", "out"}


if __name__ == "__main__":
    test_ideal_probe_model()
    test_basic_probe_insertion()
    test_probe_on_right_side_of_connection()
    test_multiple_probes()
    test_probe_in_mzi_circuit()
    test_probe_values_match_transmission()
    test_probe_error_on_non_connected_port()
    test_probe_error_on_port_conflict()
    test_probe_error_on_instance_conflict()
    test_empty_probes_dict()
    test_probe_with_none()
    print("All tests passed!")
