""" Tests for sax.utils """

import sax

from .fixtures import dc, wg, mzi


def test_get_ports(dc):
    ports = sax.get_ports(dc)
    assert ports == ("p0", "p1", "p2", "p3")


def test_rename_ports(dc):
    portmap = {
        "p0": "in0",
        "p1": "out0",
        "p2": "out1",
        "p3": "in1",
    }
    _dc = sax.rename_ports(dc, portmap)
    ports = sax.get_ports(_dc)
    assert ports == ("in0", "out0", "out1", "in1")


def test_set_params(mzi):
    params = sax.set_params(mzi.params, wl=3.0e-6)

    def _recurse_params(params, wl=3.0e-6):
        for k, v in params.items():
            if isinstance(v, dict):
                _recurse_params(v, wl=wl)
            elif k == "wl":
                assert v == wl

    _recurse_params(params, wl=3.0e-6)
    params = sax.set_params(params, "btm", wl=5.0e-6)
    _recurse_params(params.pop("btm"), wl=5.0e-6)
    _recurse_params(params, wl=3.0e-6)
