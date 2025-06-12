"""SAX netlist parsing and utilities."""

from __future__ import annotations

import os
import re
import warnings
from copy import copy, deepcopy
from functools import lru_cache, partial
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, TypedDict, cast, overload

import networkx as nx
import numpy as np
import yaml
from natsort import natsorted
from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    model_validator,
)
from pydantic import BaseModel as _BaseModel

from .utils import clean_string, hash_dict


class NetlistDict(TypedDict):
    """A dictionary representation of a netlist."""

    instances: dict
    connections: dict[str, str]
    ports: dict[str, str]
    settings: NotRequired[dict[str, Any]]


RecursiveNetlistDict = dict[str, NetlistDict]


class BaseModel(_BaseModel):
    """Base model for SAX netlists and components."""

    model_config = ConfigDict(
        extra="ignore",
        json_encoders={np.ndarray: lambda arr: np.round(arr, 12).tolist()},
    )

    def __repr__(self) -> str:
        s = super().__repr__()
        return s

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash_dict(self.model_dump())


def _validate_str(s: str, what: str = "component") -> str:
    if "," in s:
        msg = f"Invalid {what} string. Should not contain ','. Got: {s}"
        raise ValueError(msg)
    s = s.split("$")[0]
    s = clean_string(s)
    return s


ComponentStr = Annotated[str, AfterValidator(_validate_str)]


class Component(BaseModel):
    """A component in a netlist."""

    component: ComponentStr
    settings: dict[str, Any] = Field(default_factory=dict)


PortPlacement = Literal["ce", "cw", "nc", "ne", "nw", "sc", "se", "sw", "cc", "center"]


class Placement(BaseModel):
    """A placement of a component in a netlist."""

    x: str | float = 0.0
    y: str | float = 0.0
    dx: str | float = 0.0
    dy: str | float = 0.0
    rotation: float = 0.0
    mirror: bool = False
    xmin: str | float | None = None
    xmax: str | float | None = None
    ymin: str | float | None = None
    ymax: str | float | None = None
    port: str | PortPlacement | None = None


def _component_from_partial(p: partial) -> Component:
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
    return Component(component=f.__name__, settings=settings)


def _coerce_component(obj: Any) -> Component:  # noqa: ANN401
    if isinstance(obj, str):
        return Component(component=obj)
    if isinstance(obj, partial):
        return _component_from_partial(obj)
    if callable(obj):
        return _coerce_component(obj.__name__)
    if isinstance(obj, dict) and "info" in obj:
        info = obj.pop("info", {})
        settings = obj.pop("settings", {})
        obj["settings"] = {**settings, **info}
    return Component.model_validate(obj)


CoercingComponent = Annotated[Component, BeforeValidator(_coerce_component)]


_validate_instance_str = partial(_validate_str, what="instance")
_validate_port_str = partial(_validate_str, what="port")


def _validate_instance_port_str(s: str) -> str:
    parts = s.split(",")
    if len(parts) != 2:
        msg = f"Invalid instance,port string. Should contain exactly one ','. Got: {s}"
        raise ValueError(msg)
    i, p = parts
    i = _validate_instance_str(i)
    p = _validate_port_str(p)
    return f"{i},{p}"


InstanceStr = Annotated[str, AfterValidator(_validate_instance_str)]
PortStr = Annotated[str, AfterValidator(_validate_port_str)]
InstancePortStr = Annotated[str, AfterValidator(_validate_instance_port_str)]


def _nets_to_connections(
    nets: list[dict[str, str]], connections: dict[str, str]
) -> dict[str, str]:
    connections = copy(connections)
    inverse_connections = {v: k for k, v in connections.items()}

    def _is_connected(p: str) -> bool:
        return (p in connections) or (p in inverse_connections)

    def _add_connection(p: str, q: str) -> None:
        connections[p] = q
        inverse_connections[q] = p

    def _get_connected_port(p: str) -> str:
        if p in connections:
            return connections[p]
        return inverse_connections[p]

    for net in nets:
        p = net["p1"]
        q = net["p2"]
        if _is_connected(p):
            _q = _get_connected_port(p)
            msg = (
                "SAX currently does not support multiply connected ports. "
                f"Got {p}<->{q} and {p}<->{_q}"
            )
            raise ValueError(msg)
        if _is_connected(q):
            _p = _get_connected_port(q)
            msg = (
                "SAX currently does not support multiply connected ports. "
                f"Got {p}<->{q} and {_p}<->{q}"
            )
            raise ValueError(msg)
        _add_connection(p, q)
    return connections


class Netlist(BaseModel):
    """A netlist in SAX format."""

    instances: dict[InstanceStr, CoercingComponent] = Field(default_factory=dict)
    connections: dict[InstancePortStr, InstancePortStr] = Field(default_factory=dict)
    ports: dict[PortStr, InstancePortStr] = Field(default_factory=dict)
    placements: dict[InstanceStr, Placement] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)  # TODO: use this

    @model_validator(mode="before")
    @classmethod
    def coerce_nets_into_connections(cls, netlist: dict) -> dict[str, str]:
        """Convert nets into connections."""
        if not isinstance(netlist, dict):
            return netlist
        if "nets" in netlist:
            nets = netlist.pop("nets", [])
            connections = netlist.pop("connections", {})
            connections = _nets_to_connections(nets, connections)
            netlist["connections"] = connections
        return netlist


class RecursiveNetlist(RootModel):
    """A recursive netlist in SAX format."""

    root: dict[str, Netlist]

    model_config = ConfigDict(
        json_encoders={np.ndarray: lambda arr: np.round(arr, 12).tolist()},
    )

    def __repr__(self) -> str:
        s = super().__repr__()
        return s

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash_dict(self.model_dump())


AnyNetlist = dict | Netlist | NetlistDict | RecursiveNetlist | RecursiveNetlistDict


def netlist(
    netlist: Any,  # noqa: ANN401
    *,
    with_unconnected_instances: bool = True,
    with_placements: bool = True,
) -> RecursiveNetlist:
    """Return a netlist from a given dictionary."""
    if isinstance(netlist, RecursiveNetlist):
        net = netlist
    elif isinstance(netlist, Netlist):
        net = RecursiveNetlist(root={"top_level": netlist})
    elif isinstance(netlist, dict):
        if is_recursive(netlist):
            net = RecursiveNetlist.model_validate(netlist)
        else:
            flat_net = Netlist.model_validate(netlist)
            net = RecursiveNetlist.model_validate({"top_level": flat_net})
    else:
        msg = (
            "Invalid argument for `netlist`. "
            "Expected type: dict | Netlist | RecursiveNetlist. "
            f"Got: {type(netlist)}."
        )
        raise TypeError(msg)
    if not with_unconnected_instances:
        recnet_dict: RecursiveNetlistDict = _remove_unused_instances(net.model_dump())
        net = RecursiveNetlist.model_validate(recnet_dict)
    if not with_placements:
        for _net in net.root.values():
            _net.placements = {}
    return net


def flatten_netlist(recnet: RecursiveNetlistDict, sep: str = "~") -> NetlistDict:
    """Flatten a recursive netlist into a single netlist."""
    first_name = next(iter(recnet.keys()))
    net = _copy_netlist(recnet[first_name])
    _flatten_netlist(recnet, net, sep)
    return net


@lru_cache
def load_netlist(pic_path: str | Path) -> Netlist:
    """Load a netlist from a YAML file."""
    pic_path = Path(pic_path).resolve()
    net = yaml.safe_load(pic_path.read_text())
    return Netlist.model_validate(net)


@lru_cache
def load_recursive_netlist(pic_path: str | Path, ext: str = ".yml") -> RecursiveNetlist:
    """Load a recursive netlist from a folder containing YAML files."""
    pic_path = Path(pic_path).resolve()
    folder_path = pic_path.parent

    def _clean_string(path: str) -> str:
        return clean_string(re.sub(ext, "", os.path.split(path)[-1]))

    # the circuit we're interested in should come first:
    netlists: dict[str, Netlist] = {_clean_string(str(pic_path)): Netlist()}

    for filename in folder_path.iterdir():
        path = folder_path / filename
        if not path.exists() or not str(path).endswith(ext):
            continue
        netlists[_clean_string(str(path))] = load_netlist(path)

    return RecursiveNetlist.model_validate(netlists)


def is_recursive(netlist: AnyNetlist) -> bool:
    """Check if the given netlist is recursive."""
    if isinstance(netlist, RecursiveNetlist):
        return True
    if isinstance(netlist, dict):
        return "instances" not in netlist
    return False


def is_not_recursive(netlist: AnyNetlist) -> bool:
    """Check if the given netlist is not recursive."""
    return not is_recursive(netlist)


def get_netlist_instances_by_prefix(
    recursive_netlist: RecursiveNetlist,
    prefix: str,
) -> list[str]:
    """Returns a list of all instances with a given prefix in a recursive netlist.

    Args:
        recursive_netlist: The recursive netlist to search.
        prefix: The prefix to search for.

    Returns:
        A list of all instances with the given prefix.
    """
    recursive_netlist_root = recursive_netlist.model_dump()
    result = []
    for key in recursive_netlist_root:
        if key.startswith(prefix):
            result.append(key)  # noqa: PERF401
    return result


def get_component_instances(
    recursive_netlist: RecursiveNetlist,
    top_level_prefix: str,
    component_name_prefix: str,
) -> dict[str, list[str]]:
    """Get all instances of a given component in a recursive netlist.

    Args:
        recursive_netlist: The recursive netlist to search.
        top_level_prefix: The prefix of the top level instance.
        component_name_prefix: The name of the component to search for.

    Returns:
        A dictionary of all instances of the given component.
    """
    instance_names = []
    recursive_netlist_root = recursive_netlist.model_dump()

    # Should only be one in a netlist-to-digraph. Can always be very specified.
    top_level_prefixes = get_netlist_instances_by_prefix(
        recursive_netlist, prefix=top_level_prefix
    )
    top_level_prefix = top_level_prefixes[0]
    for key in recursive_netlist_root[top_level_prefix]["instances"]:
        if recursive_netlist_root[top_level_prefix]["instances"][key][
            "component"
        ].startswith(component_name_prefix):
            # Note priority encoding on match.
            instance_names.append(key)  # noqa: PERF401
    return {component_name_prefix: instance_names}


def _remove_unused_instances(
    recursive_netlist: RecursiveNetlistDict,
) -> RecursiveNetlistDict:
    recursive_netlist = {**recursive_netlist}

    for name, flat_netlist in recursive_netlist.items():
        recursive_netlist[name] = _remove_unused_instances_flat(flat_netlist)

    return recursive_netlist


def _get_connectivity_netlist(netlist: NetlistDict) -> dict[str, Any]:
    connectivity_netlist = {
        "instances": natsorted(netlist["instances"]),
        "connections": [
            (c1.split(",")[0], c2.split(",")[0])
            for c1, c2 in netlist["connections"].items()
        ],
        "ports": [(p, c.split(",")[0]) for p, c in netlist["ports"].items()],
    }
    return connectivity_netlist


def _get_connectivity_graph(netlist: NetlistDict) -> nx.Graph:
    graph = nx.Graph()
    connectivity_netlist = _get_connectivity_netlist(netlist)
    for name in connectivity_netlist["instances"]:
        graph.add_node(name)
    for c1, c2 in connectivity_netlist["connections"]:
        graph.add_edge(c1, c2)
    for c1, c2 in connectivity_netlist["ports"]:
        graph.add_edge(c1, c2)
    return graph


def _get_nodes_to_remove(graph: nx.Graph, netlist: NetlistDict) -> list[str]:
    nodes = set()
    for port in netlist["ports"]:
        nodes |= nx.descendants(graph, port)
    nodes_to_remove = set(graph.nodes) - nodes
    return list(nodes_to_remove)


def _remove_unused_instances_flat(flat_netlist: NetlistDict) -> NetlistDict:
    flat_netlist = {**flat_netlist}

    flat_netlist["instances"] = {**flat_netlist["instances"]}
    flat_netlist["connections"] = {**flat_netlist["connections"]}
    flat_netlist["ports"] = {**flat_netlist["ports"]}

    graph = _get_connectivity_graph(flat_netlist)
    names = set(_get_nodes_to_remove(graph, flat_netlist))

    for name in list(names):
        if name in flat_netlist["instances"]:
            del flat_netlist["instances"][name]

    for conn1, conn2 in list(flat_netlist["connections"].items()):
        for conn in [conn1, conn2]:
            name, _ = conn.split(",")
            if name in names and conn1 in flat_netlist["connections"]:
                del flat_netlist["connections"][conn1]

    for pname, conn in list(flat_netlist["ports"].items()):
        name, _ = conn.split(",")
        if name in names and pname in flat_netlist["ports"]:
            del flat_netlist["ports"][pname]

    return flat_netlist


def _copy_netlist(net: NetlistDict) -> NetlistDict:
    new = {
        k: deepcopy(v)
        for k, v in net.items()
        if k in ["instances", "connections", "ports"]
    }
    return cast(NetlistDict, new)


def _flatten_netlist(recnet: RecursiveNetlistDict, net: NetlistDict, sep: str) -> None:  # noqa: PLR0912,C901
    for name, instance in list(net["instances"].items()):
        component = instance["component"]
        if component not in recnet:
            continue
        del net["instances"][name]
        child_net = _copy_netlist(recnet[component])
        _flatten_netlist(recnet, child_net, sep)
        for iname, iinstance in child_net["instances"].items():
            net["instances"][f"{name}{sep}{iname}"] = iinstance
        ports = {k: f"{name}{sep}{v}" for k, v in child_net["ports"].items()}
        for ip1, ip2 in list(net["connections"].items()):
            n1, p1 = ip1.split(",")
            n2, p2 = ip2.split(",")
            if n1 == name:
                del net["connections"][ip1]
                if p1 not in ports:
                    warnings.warn(
                        f"Port {ip1} not found. Connection {ip1}<->{ip2} ignored.",
                        stacklevel=2,
                    )
                    continue
                net["connections"][ports[p1]] = ip2
            elif n2 == name:
                if p2 not in ports:
                    warnings.warn(
                        f"Port {ip2} not found. Connection {ip1}<->{ip2} ignored.",
                        stacklevel=2,
                    )
                    continue
                net["connections"][ip1] = ports[p2]
        for ip1, ip2 in child_net["connections"].items():
            net["connections"][f"{name}{sep}{ip1}"] = f"{name}{sep}{ip2}"
        for p, ip2 in list(net["ports"].items()):
            try:
                n2, p2 = ip2.split(",")
            except ValueError:
                warnings.warn(
                    f"Unconventional port definition ignored: {p}->{ip2}.", stacklevel=2
                )
                continue
            if n2 == name:
                if p2 in ports:
                    net["ports"][p] = ports[p2]
                else:
                    del net["ports"][p]


@overload
def rename_instances(netlist: Netlist, mapping: dict[str, str]) -> Netlist: ...


@overload
def rename_instances(
    netlist: RecursiveNetlist, mapping: dict[str, str]
) -> RecursiveNetlist: ...


@overload
def rename_instances(netlist: NetlistDict, mapping: dict[str, str]) -> NetlistDict: ...


@overload
def rename_instances(
    netlist: RecursiveNetlistDict, mapping: dict[str, str]
) -> RecursiveNetlistDict: ...


def rename_instances(
    netlist: Netlist | RecursiveNetlist | NetlistDict | RecursiveNetlistDict,
    mapping: dict[str, str],
) -> Netlist | RecursiveNetlist | NetlistDict | RecursiveNetlistDict:
    """Rename instances in a netlist according to a mapping."""
    given_as_dict = isinstance(netlist, dict)

    if is_recursive(netlist):
        netlist = RecursiveNetlist.model_validate(netlist)
    else:
        netlist = Netlist.model_validate(netlist)

    if isinstance(netlist, RecursiveNetlist):
        net = RecursiveNetlist(
            **{
                k: rename_instances(v, mapping).model_dump()
                for k, v in netlist.root.items()
            }
        )
        return net if not given_as_dict else net.model_dump()

    # it's a sax.Netlist now:
    inverse_mapping = {v: k for k, v in mapping.items()}
    if len(inverse_mapping) != len(mapping):
        msg = "Duplicate names to map onto found."
        raise ValueError(msg)
    instances = {mapping.get(k, k): v for k, v in netlist.instances.items()}
    connections = {}
    for ip1, ip2 in netlist.connections.items():
        i1, p1 = ip1.split(",")
        i2, p2 = ip2.split(",")
        i1 = mapping.get(i1, i1)
        i2 = mapping.get(i2, i2)
        connections[f"{i1},{p1}"] = f"{i2},{p2}"
    ports = {}
    for q, ip in netlist.ports.items():
        i, p = ip.split(",")
        i = mapping.get(i, i)
        ports[q] = f"{i},{p}"

    placements = {mapping.get(k, k): v for k, v in netlist.placements.items()}
    net = Netlist(
        instances=instances,
        connections=connections,
        ports=ports,
        placements=placements,
        settings=netlist.settings,
    )
    return net if not given_as_dict else net.model_dump()


@overload
def rename_models(netlist: Netlist, mapping: dict[str, str]) -> Netlist: ...


@overload
def rename_models(
    netlist: RecursiveNetlist, mapping: dict[str, str]
) -> RecursiveNetlist: ...


@overload
def rename_models(netlist: NetlistDict, mapping: dict[str, str]) -> NetlistDict: ...


@overload
def rename_models(
    netlist: RecursiveNetlistDict, mapping: dict[str, str]
) -> RecursiveNetlistDict: ...


def rename_models(
    netlist: Netlist | RecursiveNetlist | NetlistDict | RecursiveNetlistDict,
    mapping: dict[str, str],
) -> Netlist | RecursiveNetlist | NetlistDict | RecursiveNetlistDict:
    """Rename models in a netlist according to a mapping."""
    given_as_dict = isinstance(netlist, dict)

    if is_recursive(netlist):
        netlist = RecursiveNetlist.model_validate(netlist)
    else:
        netlist = Netlist.model_validate(netlist)

    if isinstance(netlist, RecursiveNetlist):
        net = RecursiveNetlist(
            **{
                k: rename_models(v, mapping).model_dump()
                for k, v in netlist.root.items()
            }
        )
        return net if not given_as_dict else net.model_dump()

    # it's a sax.Netlist now:
    inverse_mapping = {v: k for k, v in mapping.items()}
    if len(inverse_mapping) != len(mapping):
        msg = "Duplicate names to map onto found."
        raise ValueError(msg)

    instances = {}
    for k, instance in netlist.instances.items():
        given_as_str = False
        if isinstance(instance, str):
            given_as_str = True
            instance = {
                "component": instance,
                "settings": {},
            }
        elif isinstance(instance, Component):
            instance = instance.model_dump()

        if not isinstance(instance, dict):
            msg = (
                "Expected instance to be a dictionary or a Component. "
                f"Got: {type(instance)}."
            )
            raise TypeError(msg)

        instance["component"] = mapping.get(
            instance["component"], instance["component"]
        )
        if given_as_str:
            instances[k] = instance["component"]
        else:
            instances[k] = instance

    net = Netlist(
        instances=instances,
        connections=netlist.connections,
        ports=netlist.ports,
        placements=netlist.placements,
        settings=netlist.settings,
    )
    return net if not given_as_dict else net.model_dump()
