from sax.netlist import Netlist, RecursiveNetlist


def test_empty_netlist():
    assert Netlist()


def test_coercing_instances():
    assert Netlist.model_validate({"instances": {"mmi": "mmi"}})


def test_recursive_netlist():
    net = Netlist.model_validate({"instances": {"mmi": "mmi"}})
    recnet = RecursiveNetlist.model_validate({"net": net})
    assert recnet
