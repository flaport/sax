from functools import partial

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

SAMPLE_NETLIST2 = {
    "instances": {
        "lft": sax.models.coupler,
        "top": partial(sax.models.straight, length=10.0),
        "btm": partial(sax.models.straight, length=30.0),
        "rgt": sax.models.coupler,
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


def test_empty_netlist():
    net = sax.Netlist()
    assert isinstance(net, sax.Netlist)


def test_coercing_instances():
    net = sax.Netlist.model_validate({"instances": {"mmi": "mmi"}})
    assert isinstance(net, sax.Netlist)


def test_netlist():
    recnet = sax.Netlist.model_validate(SAMPLE_NETLIST)
    assert isinstance(recnet, sax.Netlist)


def test_recursive_netlist():
    recnet = sax.RecursiveNetlist.model_validate({"top_level": SAMPLE_NETLIST})
    assert isinstance(recnet, sax.RecursiveNetlist)


def test_netlist_function_from_dict():
    net = sax.netlist(SAMPLE_NETLIST)
    assert isinstance(net, sax.RecursiveNetlist)
    assert "top_level" in net.root


def test_netlist_function_from_netlist():
    net = sax.netlist(sax.Netlist.model_validate(SAMPLE_NETLIST))
    assert isinstance(net, sax.RecursiveNetlist)
    assert "top_level" in net.root


def test_netlist_function_from_recursive_netlist():
    net = sax.netlist(
        sax.RecursiveNetlist.model_validate({"top_level": SAMPLE_NETLIST})
    )
    assert isinstance(net, sax.RecursiveNetlist)
    assert "top_level" in net.root


def test_netlist_using_instance_functions():
    net = sax.netlist(SAMPLE_NETLIST2)
    print(net)
    assert isinstance(net, sax.RecursiveNetlist)


if __name__ == "__main__":
    test_netlist_using_instance_functions()
