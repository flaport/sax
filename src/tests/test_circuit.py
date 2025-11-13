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


if __name__ == "__main__":
    print(test_circuit())
    print(test_1port_circuit())

