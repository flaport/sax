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


if __name__ == "__main__":
    print(test_circuit())
