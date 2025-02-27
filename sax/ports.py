"""Port parsing utilities."""

from typing import overload

from natsort import natsorted
from pydantic import validate_call

import sax


@overload
def get_ports(S: sax.STypeSM) -> tuple[sax.Port, ...]: ...


@overload
def get_ports(S: sax.STypeMM) -> tuple[sax.PortMode, ...]: ...


def get_ports(S: sax.SType) -> tuple[sax.Port, ...] | tuple[sax.PortMode]:
    """Get port names of a model or an stype."""
    if callable(S):
        msg = (
            "Getting the ports of a model is no longer supported. "
            "Please Evaluate the model first: Use get_ports(model()) in stead of "
            f"get_ports(model). Got: {S}"
        )
        raise TypeError(msg)
    if (sdict := sax.try_into[sax.SDict](S)) is not None:
        ports_set = {p1 for p1, _ in sdict} | {p2 for _, p2 in sdict}
        return tuple(natsorted(ports_set))
    if (with_pm := sax.try_into[sax.SCoo | sax.SDense](S)) is not None:
        *_, pm = with_pm
        return tuple(natsorted(pm.keys()))

    msg = f"Expected an SType. Got: {S!r} [{type(S)}]"
    raise TypeError(msg)


@validate_call
def get_modes(S: sax.STypeMM) -> tuple[sax.Mode, ...]:
    """Get the modes in a multimode S-matrix."""
    return tuple(get_mode(pm) for pm in get_ports(S))


@validate_call
def get_mode(pm: sax.PortMode) -> sax.Mode:
    """Get the mode from a port@mode string."""
    return pm.split("@")[1]
