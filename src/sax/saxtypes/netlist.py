"""SAX netlist type definitions.

This module defines the type system for optical circuit netlists, including
component instances, connections, ports, and placements. It provides both
flat and hierarchical netlist representations.
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Annotated, Any, Literal, NotRequired, TypeAlias

from typing_extensions import TypedDict

from sax.saxtypes.core import Name, bval, val, val_name
from sax.saxtypes.into import into, try_into
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


def extract_fields(dct: dict[str, Any], *, fields: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in dct.items() if k in fields}


Component: TypeAlias = Annotated[str, val(val_name, name="Component")]
"""The name of a component model (must be a valid Python identifier)."""


def val_instance(obj: Any) -> Instance:
    """Validate and normalize an instance definition.

    Accepts various formats: strings, callables, dictionaries, or partial functions.
    Converts them to a standardized instance dictionary format.

    Args:
        obj: The object to validate as an instance definition.

    Returns:
        A validated instance dictionary.

    Raises:
        TypeError: If the object cannot be converted to a valid instance.

    Examples:
        Validate python objects into an instance definition:

        ```python
        import sax.saxtypes.netlist as nl

        # String component name
        inst = nl.val_instance("coupler")
        # {"component": "coupler"}

        # Dictionary with settings
        inst = nl.val_instance({"component": "coupler", "settings": {"coupling": 0.5}})

        # Partial function
        from functools import partial

        inst = nl.val_instance(partial(coupler_model, coupling=0.5))
        ```
    """
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
        array = obj.get("array", None)
        component = obj["component"]
        settings = {**obj.get("settings", {})}
        if "info" in obj:
            settings.update(obj.get("info", {}))
        inst: Instance = {
            "component": component,
            "settings": settings,
        }
        if isinstance(array, dict) and "columns" in array and "rows" in array:
            inst["array"] = {
                "columns": int(array["columns"]),
                "rows": int(array["rows"]),
            }
            if "column_pitch" in array:
                inst["array"]["column_pitch"] = float(array["column_pitch"])
            if "row_pitch" in array:
                inst["array"]["row_pitch"] = float(array["row_pitch"])
        return inst
    msg = f"Cannot coerce {obj} [{type(obj)}] into a component dictionary."
    raise TypeError(msg)


ArrayConfig = Annotated[
    TypedDict(
        "ArrayConfig",
        {
            "columns": int,
            "rows": int,
            "column_pitch": NotRequired[float],
            "row_pitch": NotRequired[float],
        },
    ),
    bval(extract_fields, fields=("columns", "rows", "column_pitch", "row_pitch")),
]
"""Configuration for arrayed component instances.

Attributes:
    columns: Number of columns in the array.
    rows: Number of rows in the array.
    column_pitch: Optional spacing between columns.
    row_pitch: Optional spacing between rows.
"""

Instance = Annotated[
    TypedDict(
        "Instance",
        {
            "component": Component,
            "settings": NotRequired[Settings],
            "array": NotRequired[ArrayConfig],
        },
    ),
    val(val_instance),
]
"""An component instantiation in a netlist with optional settings and array config.

Attributes:
    component: The name of the model.
    settings: Optional settings for the instance.
    array: Optional configuration for arrayed instances.
"""

Instances: TypeAlias = dict[InstanceName, Instance]
"""A mapping from instance names to their definitions."""

Connections: TypeAlias = dict[InstancePort, InstancePort]
"""A mapping defining point-to-point connections between instance ports."""


def val_ports(obj: Any) -> Ports:
    """Validate a ports definition for a netlist.

    Ensures that at least two ports are defined (minimum for a meaningful circuit).

    Args:
        obj: The object to validate as a ports definition.

    Returns:
        The validated ports mapping.

    Raises:
        TypeError: If fewer than two ports are defined.

    Examples:
        Validate a ports definition for a netlist:

        ```python
        import sax.saxtypes.netlist as nl

        # Valid ports definition
        ports = nl.val_ports({"in": "coupler1,in0", "out": "coupler1,out0"})
        ```
    """
    from .into import into

    ports: dict[str, InstancePort] = into[dict[str, InstancePort]](obj)
    if len(ports) < 2:
        msg = "A sax netlist needs to have at least two ports defined."
        raise TypeError(msg)
    return ports


Ports: TypeAlias = Annotated[dict[Port, InstancePort], val(val_ports)]
"""A mapping from external circuit ports to internal instance ports."""

_PortPlacement: TypeAlias = Literal[
    "ce", "cw", "nc", "ne", "nw", "sc", "se", "sw", "cc", "center"
]


Placement = Annotated[
    TypedDict(
        "Placement",
        {
            "x": str | float,
            "y": str | float,
            "dx": NotRequired[str | float],
            "dy": NotRequired[str | float],
            "rotation": NotRequired[float],
            "mirror": NotRequired[bool],
            "xmin": NotRequired[str | float | None],
            "xmax": NotRequired[str | float | None],
            "ymin": NotRequired[str | float | None],
            "ymax": NotRequired[str | float | None],
            "port": NotRequired[str | _PortPlacement | None],
        },
    ),
    bval(
        extract_fields,
        fields=(
            "x",
            "y",
            "dx",
            "dy",
            "rotation",
            "mirror",
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "port",
        ),
    ),
]
"""Physical placement information for an instance.

Defines the position, orientation, and constraints for placing
an instance in physical layout coordinates.

Attributes:
    x: X coordinate position.
    y: Y coordinate position.
    dx: Optional X offset.
    dy: Optional Y offset.
    rotation: Optional rotation angle in degrees.
    mirror: Optional mirroring flag.
    xmin: Optional minimum X constraint.
    xmax: Optional maximum X constraint.
    ymin: Optional minimum Y constraint.
    ymax: Optional maximum Y constraint.
    port: Optional port anchor specification.
"""


Placements: TypeAlias = dict[InstanceName, Placement]
"""A mapping from instance names to their physical placements."""


Net = Annotated[
    TypedDict(
        "Net",
        {
            "p1": InstancePort,
            "p2": InstancePort,
            "settings": NotRequired[Settings],
            "name": NotRequired[str | None],
        },
    ),
    bval(extract_fields, fields=("p1", "p2", "settings", "name")),
]
"""A logical connection between two ports.

Represents a point-to-point connection with optional metadata.

Attributes:
    p1: First port in the connection.
    p2: Second port in the connection.
    settings: Optional connection settings.
    name: Optional connection name.
"""


Nets: TypeAlias = list[Net]
"""A list of logical connections between ports."""


def val_netlist(obj: Any) -> dict:
    if not isinstance(obj, dict):
        msg = f"Expected a dictionary for recursive netlist, got {type(obj)}."
        raise TypeError(msg)

    obj = {**obj}

    nets = list(obj.pop("nets", []))

    if "routes" in obj:
        for bundle_name, bundle in obj["routes"].items():
            if not isinstance(bundle, dict):
                msg = f"Expected a dictionary for routes, got {type(bundle)}."
                raise TypeError(msg)
            if "links" not in bundle:
                msg = "Each route bundle must contain a 'links' key."
                raise TypeError(msg)
            for p1, p2 in bundle["links"].items():
                p1 = into[InstancePort](p1)
                p2 = into[InstancePort](p2)
                nets.append({"p1": p1, "p2": p2, "name": bundle_name})

    obj["nets"] = nets
    fields = (
        "instances",
        "connections",
        "ports",
        "nets",
        "placements",
        "settings",
    )
    return extract_fields(obj, fields=fields)


Netlist = Annotated[
    TypedDict(
        "Netlist",
        {
            "instances": Instances,
            "connections": NotRequired[Connections],
            "ports": Ports,
            "nets": NotRequired[Nets],
            "placements": NotRequired[Placements],
            "settings": NotRequired[Settings],
        },
    ),
    bval(val_netlist),
]
"""A complete netlist definition for an optical circuit.

Contains all information needed to define a circuit: instances,
connections, external ports, and optional placement/settings.

Attributes:
    instances: The component instances in the circuit.
    connections: Point-to-point connections between instances.
    ports: Mapping of external ports to internal instance ports.
    nets: Alternative connection specification as a list.
    placements: Physical placement information for instances.
    settings: Global circuit settings.
"""


def val_recnet(obj: Any) -> RecursiveNetlist:
    if not isinstance(obj, dict):
        msg = f"Expected a dictionary for recursive netlist, got {type(obj)}."
        raise TypeError(msg)

    net = try_into[Netlist](obj)
    if net is not None:
        msg = f"Expected a recursive netlist, got a flat netlist: {net}."
        raise TypeError(msg)

    ret = {}
    for name, netlist in obj.items():
        name = into[Name](name)
        net = try_into[Netlist](netlist)
        if net is None:
            msg = (
                f"Could not validate netlist for {name!r}. "
                "This netlist will be ignored."
            )
            warnings.warn(msg, stacklevel=2)
            continue
        if len(net.get("ports", {})) < 2:
            msg = (
                f"Netlist {name!r} has fewer than two ports defined. "
                "This netlist will be ignored."
            )
            warnings.warn(msg, stacklevel=2)
            continue
        ret[name] = net
    return ret


RecursiveNetlist: TypeAlias = Annotated[dict[Name, Netlist], val(val_recnet)]
"""A hierarchical netlist containing multiple named circuits."""

AnyNetlist: TypeAlias = Netlist | RecursiveNetlist | dict[str, dict[str, str]]
"""Any valid netlist format: flat, recursive, or simplified dictionary."""


def _instance_from_partial(p: partial) -> Instance:
    """Convert a partial function to an instance definition.

    Extracts the component name and keyword arguments from a partial function
    to create a standardized instance dictionary.

    Args:
        p: The partial function to convert.

    Returns:
        An instance dictionary.

    Raises:
        ValueError: If the partial has positional arguments.
        TypeError: If the partial is invalid.

    Examples:
        Validate a partial function into an instance definition:

        ```python
        from functools import partial

        # Create partial with settings
        partial_model = partial(coupler_model, coupling=0.5, loss=0.1)
        instance = _instance_from_partial(partial_model)
        # {"component": "coupler_model", "settings": {"coupling": 0.5, "loss": 0.1}}
        ```
    """
    settings: Settings = {}
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
