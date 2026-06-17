import sax

SAMPLE_NETLIST = {
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


def test_circuit() -> None:
    _mzi, _info = sax.circuit(
        netlist=SAMPLE_NETLIST,
        models={
            "coupler": sax.models.coupler_ideal,
            "waveguide": sax.models.straight,
        },
    )


def test_1port_circuit() -> None:
    """Test that 1-port circuits are supported."""
    netlist = {
        "instances": {
            "wg1": "waveguide",
        },
        "connections": {},
        "ports": {
            "in": "wg1,in0",
        },
    }

    models = {
        "waveguide": sax.models.straight,
    }

    circuit, _info = sax.circuit(netlist, models)
    result = circuit()

    # Verify the circuit has exactly 1 port
    ports = sax.get_ports(result)
    assert len(ports) == 1
    assert "in" in ports

    # Verify the S-matrix has the expected structure
    assert ("in", "in") in result


def test_circuit_with_portless_subnetlist() -> None:
    """Test that a sub-netlist without a 'ports' key is filtered out (#95)."""
    netlist = {
        "top_level": {
            "instances": {
                "wg": {"component": "waveguide"},
            },
            "connections": {},
            "ports": {
                "in": "wg,in0",
                "out": "wg,out0",
            },
        },
        # Sub-netlist with no "ports" key at all
        "unused_sub": {
            "instances": {
                "x": {"component": "waveguide"},
            },
            "connections": {},
        },
    }

    models = {"waveguide": sax.models.straight}
    circuit, _ = sax.circuit(netlist, models)
    result = circuit()
    assert set(sax.get_ports(result)) == {"in", "out"}


def test_circuit_with_empty_ports_subnetlist() -> None:
    """Test that a sub-netlist with 'ports': {} is filtered out (#95)."""
    netlist = {
        "top_level": {
            "instances": {
                "wg": {"component": "waveguide"},
            },
            "connections": {},
            "ports": {
                "in": "wg,in0",
                "out": "wg,out0",
            },
        },
        # Sub-netlist with empty ports dict
        "unused_sub": {
            "instances": {
                "x": {"component": "waveguide"},
            },
            "connections": {},
            "ports": {},
        },
    }

    models = {"waveguide": sax.models.straight}
    circuit, _ = sax.circuit(netlist, models)
    result = circuit()
    assert set(sax.get_ports(result)) == {"in", "out"}


def test_numbered_component_variants() -> None:
    """Numbered component variants from recursive netlists resolve to the base model.

    gdsfactory's CountedNetlistNamer appends numeric suffixes (coupler, coupler2, ...)
    when exporting recursive netlists. All variants should resolve to the base model.
    See https://github.com/gdsfactory/sax/issues/120.
    """
    netlist = {
        "top_level": {
            "instances": {
                "lft": {"component": "coupler"},
                "top": {"component": "waveguide"},
                "btm": {"component": "waveguide2"},
                "rgt": {"component": "coupler2"},
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
        "coupler": {
            "instances": {
                "wg": {"component": "waveguide"},
            },
            "connections": {},
            "ports": {"in0": "wg,in0", "out0": "wg,out0"},
        },
        "coupler2": {
            "instances": {
                "wg": {"component": "waveguide"},
            },
            "connections": {},
            "ports": {"in0": "wg,in0", "out0": "wg,out0"},
        },
    }

    models = {
        "coupler": sax.models.coupler_ideal,
        "waveguide": sax.models.straight,
    }

    circuit_fn, _info = sax.circuit(netlist, models)
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "in0" in ports
    assert "out0" in ports


def test_numbered_variant_explicit_model_takes_priority() -> None:
    """When an explicit model exists for a numbered variant, use it directly."""
    netlist = {
        "instances": {
            "wg1": {"component": "waveguide"},
            "wg2": {"component": "waveguide2"},
        },
        "connections": {
            "wg1,out0": "wg2,in0",
        },
        "ports": {
            "in": "wg1,in0",
            "out": "wg2,out0",
        },
    }

    def custom_waveguide2(**kwargs: sax.SettingsValue) -> sax.SType:
        return sax.models.straight(**kwargs)

    models = {
        "waveguide": sax.models.straight,
        "waveguide2": custom_waveguide2,
    }

    circuit_fn, _info = sax.circuit(netlist, models)
    result = circuit_fn()
    ports = sax.get_ports(result)
    assert "in" in ports
    assert "out" in ports


if __name__ == "__main__":
    print(test_circuit())
    print(test_1port_circuit())
    print(test_circuit_with_portless_subnetlist())
    print(test_circuit_with_empty_ports_subnetlist())
    print(test_numbered_component_variants())
    print(test_numbered_variant_explicit_model_takes_priority())
