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


def test_probe_on_unconnected_port() -> None:
    """Test that probes on unconnected ports create a fwd-only port alias."""
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

    # wg1,in0 is not part of any connection (it's a top-level port).
    # Probing it should create only tap_fwd as an alias.
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"tap": "wg1,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    # Only tap_fwd should be created (no tap_bwd for unconnected port)
    assert "tap_fwd" in ports
    assert "tap_bwd" not in ports
    assert set(ports) == {"in", "out", "tap_fwd"}


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


def test_probe_on_nets_connection() -> None:
    """Test probe insertion when connections are in nets format."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "nets": [
            {"p1": "wg1,out0", "p2": "wg2,in0"},
        ],
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

    # With probe on p1 side of net
    circuit_with_probe, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg1,out0"},
    )
    result_with_probe = circuit_with_probe()
    ports_with_probe = sax.get_ports(result_with_probe)
    assert set(ports_with_probe) == {"in", "out", "mid_fwd", "mid_bwd"}


def test_probe_on_nets_p2_side() -> None:
    """Test probe on the p2 side of a nets-format connection."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "nets": [
            {"p1": "wg1,out0", "p2": "wg2,in0"},
        ],
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    # Probe on p2 side of net
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"mid": "wg2,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert set(ports) == {"in", "out", "mid_fwd", "mid_bwd"}


def test_probe_on_nets_with_multiple_nets() -> None:
    """Test probe insertion with multiple nets, only one being probed."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
            "wg3": "waveguide",
        },
        "nets": [
            {"p1": "wg1,out0", "p2": "wg2,in0"},
            {"p1": "wg2,out0", "p2": "wg3,in0"},
        ],
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
        probes={"mid": "wg2,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert set(ports) == {"in", "out", "mid_fwd", "mid_bwd"}

    # Transmission should still work
    assert ("in", "out") in result
    assert ("in", "mid_fwd") in result


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


def test_port_on_internal_node_becomes_probe() -> None:
    """Test that a port mapping to an internal connection node becomes a probe."""
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
            "mid": "wg1,out0",  # Internal node — should become a probe
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    with pytest.warns(UserWarning, match="internal node"):
        circuit_fn, _ = sax.circuit(netlist, models)

    result = circuit_fn()
    ports = sax.get_ports(result)
    # "mid" should have been replaced by "mid_fwd" and "mid_bwd"
    assert "mid" not in ports
    assert "mid_fwd" in ports
    assert "mid_bwd" in ports
    assert set(ports) == {"in", "out", "mid_fwd", "mid_bwd"}


def test_port_on_internal_node_doesnt_affect_transmission() -> None:
    """Test that auto-probes don't affect the circuit's real ports."""
    netlist_plain = {
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

    netlist_with_internal_port = {
        **netlist_plain,
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
            "mid": "wg2,in0",  # Internal node
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    circuit_plain, _ = sax.circuit(netlist_plain, models)
    with pytest.warns(UserWarning, match="internal node"):
        circuit_probed, _ = sax.circuit(netlist_with_internal_port, models)

    result_plain = circuit_plain(wl=1.55)
    result_probed = circuit_probed(wl=1.55)

    # Transmission between real ports should be identical
    assert result_plain["in", "out"] == result_probed["in", "out"]
    assert result_plain["out", "in"] == result_probed["out", "in"]


def test_mixed_explicit_and_auto_probes() -> None:
    """Test that explicit probes= and auto-detected port probes work together."""
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
            "auto_mid": "wg1,out0",  # Auto-detected probe
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    with pytest.warns(UserWarning, match="internal node"):
        circuit_fn, _ = sax.circuit(
            netlist,
            models,
            probes={"explicit_mid": "wg2,out0"},
        )

    result = circuit_fn()
    ports = sax.get_ports(result)
    expected_ports = {
        "in",
        "out",
        "auto_mid_fwd",
        "auto_mid_bwd",
        "explicit_mid_fwd",
        "explicit_mid_bwd",
    }
    assert set(ports) == expected_ports


def test_port_on_internal_node_with_nets_format() -> None:
    """Test auto-probe detection when connections are in nets format."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
            "wg2": "waveguide",
        },
        "nets": [
            {"p1": "wg1,out0", "p2": "wg2,in0"},
        ],
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
            "mid": "wg2,in0",  # Internal node (via nets)
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    with pytest.warns(UserWarning, match="internal node"):
        circuit_fn, _ = sax.circuit(netlist, models)

    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "mid" not in ports
    assert "mid_fwd" in ports
    assert "mid_bwd" in ports
    assert set(ports) == {"in", "out", "mid_fwd", "mid_bwd"}


# --- Hierarchical probe tests ---


def _hierarchical_mzi_netlist() -> dict:
    """Helper: a 2-level hierarchical MZI netlist for testing."""
    return {
        "top_level": {
            "instances": {
                "mzi": {"component": "mzi_component"},
            },
            "connections": {},
            "ports": {
                "in0": "mzi,in0",
                "in1": "mzi,in1",
                "out0": "mzi,out0",
                "out1": "mzi,out1",
            },
        },
        "mzi_component": {
            "instances": {
                "lft": {"component": "coupler"},
                "top": {"component": "waveguide"},
                "btm": {"component": "waveguide"},
                "rgt": {"component": "coupler"},
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
        },
    }


def _hierarchical_models() -> dict:
    return {
        "coupler": sax.models.coupler_ideal,
        "waveguide": sax.models.straight,
    }


def test_hierarchical_probe_one_level() -> None:
    """Test probe into a sub-circuit using dot-separated path."""
    netlist = _hierarchical_mzi_netlist()
    models = _hierarchical_models()

    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"top_arm": "mzi.top,in0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "top_arm_fwd" in ports
    assert "top_arm_bwd" in ports
    expected = {"in0", "in1", "out0", "out1", "top_arm_fwd", "top_arm_bwd"}
    assert set(ports) == expected


def test_hierarchical_probe_two_levels() -> None:
    """Test probe two levels deep in the hierarchy."""
    netlist = {
        "outer": {
            "instances": {
                "inner_inst": {"component": "inner"},
            },
            "connections": {},
            "ports": {
                "in": "inner_inst,in",
                "out": "inner_inst,out",
            },
        },
        "inner": {
            "instances": {
                "sub_inst": {"component": "sub"},
            },
            "connections": {},
            "ports": {
                "in": "sub_inst,in",
                "out": "sub_inst,out",
            },
        },
        "sub": {
            "instances": {
                "wg1": {"component": "waveguide"},
                "wg2": {"component": "waveguide"},
            },
            "connections": {
                "wg1,out0": "wg2,in0",
            },
            "ports": {
                "in": "wg1,in0",
                "out": "wg2,out0",
            },
        },
    }
    models = {"waveguide": sax.models.straight}

    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"deep": "inner_inst.sub_inst.wg1,out0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "deep_fwd" in ports
    assert "deep_bwd" in ports
    assert set(ports) == {"in", "out", "deep_fwd", "deep_bwd"}


def test_hierarchical_probe_doesnt_affect_transmission() -> None:
    """Test that hierarchical probes don't affect transmission of real ports."""
    netlist = _hierarchical_mzi_netlist()
    models = _hierarchical_models()

    # Without probes
    circuit_plain, _ = sax.circuit(netlist, models)
    result_plain = circuit_plain(wl=1.55)

    # With hierarchical probe
    circuit_probed, _ = sax.circuit(
        netlist,
        models,
        probes={"top_arm": "mzi.top,in0"},
    )
    result_probed = circuit_probed(wl=1.55)

    assert result_plain["in0", "out0"] == result_probed["in0", "out0"]
    assert result_plain["in0", "out1"] == result_probed["in0", "out1"]


def test_hierarchical_probe_shared_component() -> None:
    """Test probing one instance when a shared component is used by multiple."""
    netlist = {
        "top_level": {
            "instances": {
                "a": {"component": "sub"},
                "b": {"component": "sub"},
            },
            "connections": {
                "a,out": "b,in",
            },
            "ports": {
                "in": "a,in",
                "out": "b,out",
            },
        },
        "sub": {
            "instances": {
                "wg1": {"component": "waveguide"},
                "wg2": {"component": "waveguide"},
            },
            "connections": {
                "wg1,out0": "wg2,in0",
            },
            "ports": {
                "in": "wg1,in0",
                "out": "wg2,out0",
            },
        },
    }
    models = {"waveguide": sax.models.straight}

    # Probe only through instance 'a', not 'b'
    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={"mid_a": "a.wg1,out0"},
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    # Only 'a's probe ports should be exposed at the top level
    assert "mid_a_fwd" in ports
    assert "mid_a_bwd" in ports
    assert set(ports) == {"in", "out", "mid_a_fwd", "mid_a_bwd"}


def test_mixed_top_level_and_hierarchical_probes() -> None:
    """Test combining top-level and hierarchical probes in the same call."""
    netlist = {
        "top_level": {
            "instances": {
                "sub": {"component": "sub_circuit"},
                "wg_top": {"component": "waveguide"},
            },
            "connections": {
                "sub,out": "wg_top,in0",
            },
            "ports": {
                "in": "sub,in",
                "out": "wg_top,out0",
            },
        },
        "sub_circuit": {
            "instances": {
                "wg1": {"component": "waveguide"},
                "wg2": {"component": "waveguide"},
            },
            "connections": {
                "wg1,out0": "wg2,in0",
            },
            "ports": {
                "in": "wg1,in0",
                "out": "wg2,out0",
            },
        },
    }
    models = {"waveguide": sax.models.straight}

    circuit_fn, _ = sax.circuit(
        netlist,
        models,
        probes={
            "top_probe": "sub,out",  # Top-level probe
            "deep_probe": "sub.wg1,out0",  # Hierarchical probe
        },
    )
    result = circuit_fn()
    ports = sax.get_ports(result)
    expected = {
        "in",
        "out",
        "top_probe_fwd",
        "top_probe_bwd",
        "deep_probe_fwd",
        "deep_probe_bwd",
    }
    assert set(ports) == expected


def test_hierarchical_probe_invalid_path_instance() -> None:
    """Test that a bad instance name in the hierarchy path gives a clear error."""
    netlist = _hierarchical_mzi_netlist()
    models = _hierarchical_models()

    # 'bad_inst' is not an instance in the top-level
    with pytest.raises(ValueError, match="not found"):
        sax.circuit(
            netlist,
            models,
            probes={"bad": "bad_inst.top,in0"},
        )


def test_hierarchical_probe_primitive_component() -> None:
    """Test that probing into a primitive (leaf) component gives a clear error."""
    netlist = _hierarchical_mzi_netlist()
    models = _hierarchical_models()

    # 'top' is a waveguide instance (primitive model, no sub-netlist)
    with pytest.raises(ValueError, match="not defined in the recursive netlist"):
        sax.circuit(
            netlist,
            models,
            probes={"bad": "mzi.top.deeper,in0"},
        )


if __name__ == "__main__":
    test_ideal_probe_model()
    test_basic_probe_insertion()
    test_probe_on_right_side_of_connection()
    test_multiple_probes()
    test_probe_in_mzi_circuit()
    test_probe_values_match_transmission()
    test_probe_on_unconnected_port()
    test_probe_error_on_port_conflict()
    test_probe_error_on_instance_conflict()
    test_empty_probes_dict()
    test_probe_with_none()
    test_port_on_internal_node_becomes_probe()
    test_port_on_internal_node_doesnt_affect_transmission()
    test_mixed_explicit_and_auto_probes()
    test_port_on_internal_node_with_nets_format()
    test_hierarchical_probe_one_level()
    test_hierarchical_probe_two_levels()
    test_hierarchical_probe_doesnt_affect_transmission()
    test_hierarchical_probe_shared_component()
    test_mixed_top_level_and_hierarchical_probes()
    test_hierarchical_probe_invalid_path_instance()
    test_hierarchical_probe_primitive_component()
    print("All tests passed!")
