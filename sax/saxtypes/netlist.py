"""SAX netlist definitions."""

from __future__ import annotations

from functools import partial
from typing import Annotated, Any, Literal, NotRequired, TypeAlias, TypedDict

from sax.saxtypes.core import Name, val, val_name
from sax.saxtypes.settings import Settings
from sax.saxtypes.singlemode import InstanceName, InstancePort, Port

__all__ = [
    "AnyNetlist",
    "Component",
    "Connections",
    "Instance",
    "Instances",
    "Net",
    "Netlist",
    "Nets",
    "Placement",
    "Placements",
    "Ports",
    "RecursiveNetlist",
]


Component: TypeAlias = Annotated[str, val(val_name, name="Component")]
"""The name of an instance component (model / cell / ...)."""


def val_instance(obj: Any) -> Instance:
    if isinstance(obj, str):
        return {"component": obj}
    if isinstance(obj, partial):
        return _instance_from_partial(obj)
    if callable(obj):
        return val_instance(obj.__name__)
    if isinstance(obj, dict):
        if "component" not in obj:
            msg = (
                f"Component dictionaries need to contain a 'component' key. Got: {obj}."
            )
            raise TypeError(msg)
        component = obj["component"]
        settings = {**obj.pop("settings", {})}
        if "info" in obj:
            settings.update(obj.pop("info", {}))
        return {
            "component": component,
            "settings": settings,
        }
    msg = f"Cannot coerce {obj} [{type(obj)}] into a component dictionary."
    raise TypeError(msg)


class _Instance(TypedDict):
    component: Component
    settings: NotRequired[Settings]


Instance: TypeAlias = Annotated[_Instance, val(val_instance)]
"""An instantiation of a cell in a netlist."""

Instances: TypeAlias = dict[InstanceName, Instance]
"""A mapping of instance names to instance definitions."""

Connections: TypeAlias = dict[InstancePort, InstancePort]
"""A mapping between connected ports."""


def val_ports(obj: Any) -> Ports:
    from .into import into

    ports: dict[str, InstancePort] = into[dict[str, InstancePort]](obj)
    if len(ports) < 2:
        msg = "A sax netlist needs to have at least two ports defined."
        raise TypeError(msg)
    return ports


Ports: TypeAlias = Annotated[dict[Port, InstancePort], val(val_ports)]
"""A mapping from circuit outport ports to instance ports."""


class Placement(TypedDict):
    """A netlist placement."""

    x: str | float
    y: str | float
    dx: NotRequired[str | float]
    dy: NotRequired[str | float]
    rotation: NotRequired[float]
    mirror: NotRequired[bool]
    xmin: NotRequired[str | float | None]
    xmax: NotRequired[str | float | None]
    ymin: NotRequired[str | float | None]
    ymax: NotRequired[str | float | None]
    port: NotRequired[str | _PortPlacement | None]


_PortPlacement: TypeAlias = Literal[
    "ce", "cw", "nc", "ne", "nw", "sc", "se", "sw", "cc", "center"
]

Placements: TypeAlias = dict[InstanceName, Placement]
""" A mapping from instance names to their placements."""


class Net(TypedDict):
    """A logical connection."""

    p1: str
    p2: str
    settings: NotRequired[dict]
    name: NotRequired[str | None]


Nets: TypeAlias = list[Net]
""" A list of logical connections."""


class Netlist(TypedDict):
    """A netlist definition."""

    instances: Instances
    connections: NotRequired[Connections]
    ports: Ports
    nets: NotRequired[Nets]
    placements: NotRequired[Placements]
    settings: NotRequired[Settings]


RecursiveNetlist: TypeAlias = dict[Name, Netlist]
"""A recursive netlist definition."""

AnyNetlist: TypeAlias = Netlist | RecursiveNetlist
"""Any kind of netlist recursive or not."""


def _instance_from_partial(p: partial) -> Instance:
    settings = {}
    f: Any = p
    while isinstance(f, partial):
        if f.args:
            msg = (
                "SAX circuits and netlists don't support partials "
                "with positional arguments."
            )
            raise ValueError(msg)
        settings = {**f.keywords, **settings}
        f = f.func
    if not callable(f):
        msg = "partial of non-callable."
        raise TypeError(msg)
    if not hasattr(f, "__name__"):
        msg = "partial of component without '.__name__' attribute."
        raise TypeError(msg)
    if settings:
        return {"component": f.__name__, "settings": settings}
    return {"component": f.__name__}
