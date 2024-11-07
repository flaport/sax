""" SAX netlist parsing and utilities """

from __future__ import annotations

import os
import re
import warnings
from copy import deepcopy
from functools import lru_cache, partial
from typing import Any, Literal, TypedDict

import black
import networkx as nx
import numpy as np
import yaml
from natsort import natsorted
from pydantic import AfterValidator
from pydantic import BaseModel as _BaseModel
from pydantic import BeforeValidator, ConfigDict, Field, RootModel, model_validator
from typing_extensions import Annotated

from .utils import clean_string, hash_dict


class NetlistDict(TypedDict):
    instances: dict
    connections: dict[str, str]
    ports: dict[str, str]
    settings: dict[str, Any]


RecursiveNetlistDict = dict[str, NetlistDict]


class BaseModel(_BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        json_encoders={np.ndarray: lambda arr: np.round(arr, 12).tolist()},
    )

    def __repr__(self):
        s = super().__repr__()
        s = black.format_str(s, mode=black.Mode())
        return s

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash_dict(self.model_dump())


def _validate_str(s: str, what="component"):
    if "," in s:
        raise ValueError(f"Invalid {what} string. Should not contain ','. Got: {s}")
    s = s.split("$")[0]
    s = clean_string(s)
    return s


ComponentStr = Annotated[str, AfterValidator(_validate_str)]


class Component(BaseModel):
    component: ComponentStr
    settings: dict[str, Any] = Field(default_factory=dict)


PortPlacement = Literal["ce", "cw", "nc", "ne", "nw", "sc", "se", "sw", "cc", "center"]


class Placement(BaseModel):
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


def _component_from_partial(p: partial):
    settings = {}
    f: Any = p
    while isinstance(f, partial):
        if f.args:
            raise ValueError(
                "SAX circuits and netlists don't support partials "
                "with positional arguments."
            )
        settings = {**f.keywords, **settings}
        f = f.func
    if not callable(f):
        raise ValueError("partial of non-callable.")
    return Component(component=f.__name__, settings=settings)


def _coerce_component(obj: Any) -> Component:
    if isinstance(obj, str):
        return Component(component=obj)
    elif isinstance(obj, partial):
        return _component_from_partial(obj)
    elif callable(obj):
        return _coerce_component(obj.__name__)
    elif isinstance(obj, dict) and "info" in obj:
        info = obj.pop("info", {})
        settings = obj.pop("settings", {})
        obj["settings"] = {**settings, **info}
    return Component.model_validate(obj)


CoercingComponent = Annotated[Component, BeforeValidator(_coerce_component)]


_validate_instance_str = partial(_validate_str, what="instance")
_validate_port_str = partial(_validate_str, what="port")


def _validate_instance_port_str(s: str):
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid instance,port string. Should contain exactly one ','. Got: {s}"
        )
    i, p = parts
    i = _validate_instance_str(i)
    p = _validate_port_str(p)
    return f"{i},{p}"


InstanceStr = Annotated[str, AfterValidator(_validate_instance_str)]
PortStr = Annotated[str, AfterValidator(_validate_port_str)]
InstancePortStr = Annotated[str, AfterValidator(_validate_instance_port_str)]


def _nets_to_connections(nets: list[dict], connections: dict):
    connections = {k: v for k, v in connections.items()}
    inverse_connections = {v: k for k, v in connections.items()}

    def _is_connected(p):
        return (p in connections) or (p in inverse_connections)

    def _add_connection(p, q):
        connections[p] = q
        inverse_connections[q] = p

    def _get_connected_port(p):
        if p in connections:
            return connections[p]
        else:
            return inverse_connections[p]

    for net in nets:
        p = net["p1"]
        q = net["p2"]
        if _is_connected(p):
            _q = _get_connected_port(p)
            raise ValueError(
                "SAX currently does not support multiply connected ports. "
                f"Got {p}<->{q} and {p}<->{_q}"
            )
        if _is_connected(q):
            _p = _get_connected_port(q)
            raise ValueError(
                "SAX currently does not support multiply connected ports. "
                f"Got {p}<->{q} and {_p}<->{q}"
            )
        _add_connection(p, q)
    return connections


class Netlist(BaseModel):
    instances: dict[InstanceStr, CoercingComponent] = Field(default_factory=dict)
    connections: dict[InstancePortStr, InstancePortStr] = Field(default_factory=dict)
    ports: dict[PortStr, InstancePortStr] = Field(default_factory=dict)
    placements: dict[InstanceStr, Placement] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)  # TODO: use this

    @model_validator(mode="before")
    @classmethod
    def coerce_nets_into_connections(cls, netlist: dict):
        if not isinstance(netlist, dict):
            return netlist
        if "nets" in netlist:
            nets = netlist.pop("nets", [])
            connections = netlist.pop("connections", {})
            connections = _nets_to_connections(nets, connections)
            netlist["connections"] = connections
        return netlist


class RecursiveNetlist(RootModel):
    root: dict[str, Netlist]

    model_config = ConfigDict(
        json_encoders={np.ndarray: lambda arr: np.round(arr, 12).tolist()},
    )

    def __repr__(self):
        s = super().__repr__()
        s = black.format_str(s, mode=black.Mode())
        return s

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash_dict(self.model_dump())


AnyNetlist = Netlist | NetlistDict | RecursiveNetlist | RecursiveNetlistDict


def netlist(
    netlist: Any, with_unconnected_instances: bool = True, with_placements=True
) -> RecursiveNetlist:
    """return a netlist from a given dictionary"""
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
        raise ValueError(
            "Invalid argument for `netlist`. "
            "Expected type: dict | Netlist | RecursiveNetlist. "
            f"Got: {type(netlist)}."
        )
    if not with_unconnected_instances:
        recnet_dict: RecursiveNetlistDict = _remove_unused_instances(net.model_dump())
        net = RecursiveNetlist.model_validate(recnet_dict)
    if not with_placements:
        for _net in net.root.values():
            _net.placements = {}
    return net


def flatten_netlist(recnet: RecursiveNetlistDict, sep: str = "~"):
    first_name = list(recnet.keys())[0]
    net = _copy_netlist(recnet[first_name])
    _flatten_netlist(recnet, net, sep)
    return net


@lru_cache()
def load_netlist(pic_path: str) -> Netlist:
    with open(pic_path, "r") as file:
        net = yaml.safe_load(file.read())
    return Netlist.model_validate(net)


@lru_cache()
def load_recursive_netlist(pic_path: str, ext: str = ".yml"):
    folder_path = os.path.dirname(os.path.abspath(pic_path))

    def _clean_string(path: str) -> str:
        return clean_string(re.sub(ext, "", os.path.split(path)[-1]))

    # the circuit we're interested in should come first:
    netlists: dict[str, Netlist] = {_clean_string(pic_path): Netlist()}

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if not os.path.isfile(path) or not path.endswith(ext):
            continue
        netlists[_clean_string(path)] = load_netlist(path)

    return RecursiveNetlist.model_validate(netlists)


def is_recursive(netlist: AnyNetlist):
    if isinstance(netlist, RecursiveNetlist):
        return True
    elif isinstance(netlist, dict):
        return "instances" not in netlist
    else:
        return False


def is_not_recursive(netlist: AnyNetlist):
    return not is_recursive(netlist)


def get_netlist_instances_by_prefix(
    recursive_netlist: RecursiveNetlist,
    prefix: str,
):
    """
    Returns a list of all instances with a given prefix in a recursive netlist.

    Args:
        recursive_netlist: The recursive netlist to search.
        prefix: The prefix to search for.

    Returns:
        A list of all instances with the given prefix.
    """
    recursive_netlist_root = recursive_netlist.model_dump()
    result = []
    for key in recursive_netlist_root.keys():
        if key.startswith(prefix):
            result.append(key)
    return result


def get_component_instances(
    recursive_netlist: RecursiveNetlist,
    top_level_prefix: str,
    component_name_prefix: str,
):
    """
    Returns a dictionary of all instances of a given component in a recursive netlist.

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
            instance_names.append(key)
    return {component_name_prefix: instance_names}


def _remove_unused_instances(recursive_netlist: RecursiveNetlistDict):
    recursive_netlist = {**recursive_netlist}

    for name, flat_netlist in recursive_netlist.items():
        recursive_netlist[name] = _remove_unused_instances_flat(flat_netlist)

    return recursive_netlist


def _get_connectivity_netlist(netlist):
    connectivity_netlist = {
        "instances": natsorted(netlist["instances"]),
        "connections": [
            (c1.split(",")[0], c2.split(",")[0])
            for c1, c2 in netlist["connections"].items()
        ],
        "ports": [(p, c.split(",")[0]) for p, c in netlist["ports"].items()],
    }
    return connectivity_netlist


def _get_connectivity_graph(netlist):
    graph = nx.Graph()
    connectivity_netlist = _get_connectivity_netlist(netlist)
    for name in connectivity_netlist["instances"]:
        graph.add_node(name)
    for c1, c2 in connectivity_netlist["connections"]:
        graph.add_edge(c1, c2)
    for c1, c2 in connectivity_netlist["ports"]:
        graph.add_edge(c1, c2)
    return graph


def _get_nodes_to_remove(graph, netlist):
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


def _copy_netlist(net):
    net = {
        k: deepcopy(v)
        for k, v in net.items()
        if k in ["instances", "connections", "ports"]
    }
    return net


def _flatten_netlist(recnet, net, sep):
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
                        f"Port {ip1} not found. Connection {ip1}<->{ip2} ignored."
                    )
                    continue
                net["connections"][ports[p1]] = ip2
            elif n2 == name:
                if p2 not in ports:
                    warnings.warn(
                        f"Port {ip2} not found. Connection {ip1}<->{ip2} ignored."
                    )
                    continue
                net["connections"][ip1] = ports[p2]
        for ip1, ip2 in child_net["connections"].items():
            net["connections"][f"{name}{sep}{ip1}"] = f"{name}{sep}{ip2}"
        for p, ip2 in list(net["ports"].items()):
            try:
                n2, p2 = ip2.split(",")
            except ValueError:
                warnings.warn(f"Unconventional port definition ignored: {p}->{ip2}.")
                continue
            if n2 == name:
                if p2 in ports:
                    net["ports"][p] = ports[p2]
                else:
                    del net["ports"][p]
