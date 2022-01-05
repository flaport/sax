""" Tests for sax.utils """

import sax


def test_get_ports():
    dc = sax.models.pic.coupler()
    ports = sax.utils.get_ports(dc)
    assert ports == ("in0", "out0", "out1", "in1"), print(ports)
    # assert ports == ("p0", "p1", "p2", "p3")


def test_rename_ports():
    dc = sax.models.pic.coupler()
    portmap = {
        "in0": "o1",
        "out0": "o4",
        "out1": "o3",
        "in1": "o2",
    }
    _dc = sax.rename_ports(dc, portmap)
    ports = sax.utils.get_ports(_dc)
    assert ports == ("o1", "o4", "o3", "o2"), print(ports)


if __name__ == "__main__":
    # test_get_ports()
    test_rename_ports()
