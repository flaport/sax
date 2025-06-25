"""Netlist utilities."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import cast, overload

import networkx as nx
from natsort import natsorted

import sax

__all__ = [  # noqa: RUF022
    "netlist",
    "flatten_netlist",
]


def netlist(
    netlist: sax.AnyNetlist,
    *,
    top_level_name: str = "top_level",
) -> sax.RecursiveNetlist:
    """Convert a netlist to recursive netlist format.

    Ensures that the input netlist is in recursive netlist format, which is a
    dictionary mapping component names to their netlists. If a top-level circuit
    with the specified name exists, it will be promoted to the top level.

    Args:
        netlist: Input netlist in any format (flat netlist or recursive netlist).
        top_level_name: Name to use for the top-level circuit when converting
            from flat netlist format. Defaults to "top_level".

    Returns:
        Recursive netlist dictionary with component name keys.

    Raises:
        ValueError: If the input netlist format is invalid.

    Example:
        ```python
        # Convert flat netlist to recursive format
        flat_net = {
            "instances": {"wg1": {"component": "waveguide"}},
            "ports": {"in": "wg1,in", "out": "wg1,out"},
        }
        rec_net = netlist(flat_net, top_level_name="my_circuit")
        # Result: {"my_circuit": flat_net}
        ```
    """
    if not isinstance(netlist, dict):
        msg = (
            "Invalid argument for `netlist`. "
            "Expected type: dict | Netlist | RecursiveNetlist. "
            f"Got: {type(netlist)}."
        )
        raise TypeError(msg)

    if "instances" in netlist:
        return {top_level_name: cast(sax.Netlist, netlist)}

    recnet = cast(sax.RecursiveNetlist, netlist)
    top_level = recnet.get(top_level_name, None)
    if top_level is None:
        return recnet
    return {top_level_name: top_level, **recnet}


def flatten_netlist(recnet: sax.RecursiveNetlist, sep: str = "~") -> sax.Netlist:
    """Flatten a recursive netlist into a single flat netlist.

    Converts a hierarchical (recursive) netlist into a single flat netlist by
    inlining all sub-circuits. Instance names are prefixed to avoid conflicts.

    Args:
        recnet: Recursive netlist to flatten.
        sep: Separator used for hierarchical instance naming. Defaults to "~".

    Returns:
        Single flat netlist with all hierarchies inlined.

    Example:
        ```python
        # Flatten a hierarchical netlist
        recnet = {
            "top": {
                "instances": {"sub1": {"component": "subcircuit"}},
                "ports": {"in": "sub1,in"},
            },
            "subcircuit": {
                "instances": {"wg1": {"component": "waveguide"}},
                "ports": {"in": "wg1,in"},
            },
        }
        flat = flatten_netlist(recnet)
        # Result has instances like "sub1~wg1" for the flattened hierarchy
        ```
    """
    first_name = next(iter(recnet.keys()))
    net = deepcopy(recnet[first_name])
    _flatten_netlist_into(recnet, net, sep)
    return net


@overload
def remove_unused_instances(netlist: sax.Netlist) -> sax.Netlist: ...


@overload
def remove_unused_instances(netlist: sax.RecursiveNetlist) -> sax.RecursiveNetlist: ...


def remove_unused_instances(netlist: sax.AnyNetlist) -> sax.AnyNetlist:
    if "instances" in netlist:
        net = cast(sax.Netlist, deepcopy(netlist))
        names = _get_nodes_to_remove(_get_connectivity_graph(net), net)
        _remove_instances(net, names)
        _remove_connections(net, names)
        _remove_ports(net, names)
        return net

    recnet = cast(sax.RecursiveNetlist, {**netlist})
    for name, flat_netlist in recnet.items():
        recnet[name] = remove_unused_instances(flat_netlist)

    return recnet


@overload
def rename_instances(
    netlist: sax.Netlist,
    mapping: dict[sax.InstanceName, sax.InstanceName],
) -> sax.Netlist: ...


@overload
def rename_instances(
    netlist: sax.RecursiveNetlist,
    mapping: dict[sax.InstanceName, sax.InstanceName],
) -> sax.RecursiveNetlist: ...


def rename_instances(
    netlist: sax.AnyNetlist,
    mapping: dict[sax.InstanceName, sax.InstanceName],
) -> sax.AnyNetlist:
    if (recnet := sax.try_into[sax.RecursiveNetlist](netlist)) is not None:
        return {k: rename_instances(v, mapping) for k, v in recnet.items()}

    # it's a sax.Netlist now:
    net: sax.Netlist = sax.into[sax.Netlist](netlist)
    new: sax.Netlist = {"instances": {}, "ports": {}}
    inverse_mapping = {v: k for k, v in mapping.items()}
    if len(inverse_mapping) != len(mapping):
        msg = "Duplicate names to map onto found."
        raise ValueError(msg)
    new["instances"] = {mapping.get(k, k): v for k, v in net["instances"].items()}
    new["connections"] = {}
    for ip1, ip2 in net.get("connections", {}).items():
        i1, p1 = ip1.split(",")
        i2, p2 = ip2.split(",")
        i1 = mapping.get(i1, i1)
        i2 = mapping.get(i2, i2)
        new["connections"][f"{i1},{p1}"] = f"{i2},{p2}"
    new["ports"] = {}
    for q, ip in net["ports"].items():
        i, p = ip.split(",")
        i = mapping.get(i, i)
        new["ports"][q] = f"{i},{p}"

    new["placements"] = {
        mapping.get(k, k): v for k, v in net.get("placements", {}).items()
    }
    return {**net, **new}


@overload
def rename_models(
    netlist: sax.Netlist,
    mapping: dict[sax.Name, sax.Name],
) -> sax.Netlist: ...


@overload
def rename_models(
    netlist: sax.RecursiveNetlist,
    mapping: dict[sax.Name, sax.Name],
) -> sax.RecursiveNetlist: ...


def rename_models(
    netlist: sax.AnyNetlist,
    mapping: dict[sax.Name, sax.Name],
) -> sax.AnyNetlist:
    if (recnet := sax.try_into[sax.RecursiveNetlist](netlist)) is not None:
        return {k: rename_models(v, mapping) for k, v in recnet.items()}

    # it's a sax.Netlist now:
    net: sax.Netlist = deepcopy(sax.into[sax.Netlist](netlist))
    inverse_mapping = {v: k for k, v in mapping.items()}
    if len(inverse_mapping) != len(mapping):
        msg = "Duplicate names to map onto found."
        raise ValueError(msg)

    for inst in net["instances"].values():
        inst["component"] = mapping.get(inst["component"], inst["component"])
    return net


def _remove_ports(net: sax.Netlist, names: set[sax.InstanceName]) -> None:
    for pname, conn in list(net["ports"].items()):
        name, _ = conn.split(",")
        if name in names and pname in net["ports"]:
            del net["ports"][pname]


def _remove_connections(net: sax.Netlist, names: set[sax.InstanceName]) -> None:
    net["connections"] = {**net.get("connections", {})}
    for conn1, conn2 in list(net["connections"].items()):
        for conn in [conn1, conn2]:
            name, _ = conn.split(",")
            if name in names and conn1 in net["connections"]:
                del net["connections"][conn1]
    if not net["connections"]:
        del net["connections"]


def _remove_instances(net: sax.Netlist, names: set[sax.InstanceName]) -> None:
    for name in names:
        if name in net["instances"]:
            del net["instances"][name]


def _get_nodes_to_remove(graph: nx.Graph, netlist: sax.Netlist) -> set[str]:
    nodes = set()
    for port in netlist["ports"]:
        nodes |= nx.descendants(graph, port)
    return set(graph.nodes) - nodes


def _get_connectivity_netlist(netlist: sax.Netlist) -> dict:
    return {
        "instances": natsorted(netlist["instances"]),
        "connections": [
            (c1.split(",")[0], c2.split(",")[0])
            for c1, c2 in netlist.get("connections", {}).items()
        ],
        "ports": [(p, c.split(",")[0]) for p, c in netlist["ports"].items()],
    }


def _get_connectivity_graph(netlist: sax.Netlist) -> nx.Graph:
    graph = nx.Graph()
    connectivity_netlist = _get_connectivity_netlist(netlist)
    for name in connectivity_netlist["instances"]:
        graph.add_node(name)
    for c1, c2 in connectivity_netlist["connections"]:
        graph.add_edge(c1, c2)
    for c1, c2 in connectivity_netlist["ports"]:
        graph.add_edge(c1, c2)
    return graph


def _flatten_netlist_into(  # noqa: PLR0912,C901
    recnet: sax.RecursiveNetlist, net: sax.Netlist, sep: str
) -> None:
    for name, instance in list(net["instances"].items()):
        component = instance["component"]
        if component not in recnet:
            continue
        del net["instances"][name]
        child_net = deepcopy(recnet[component])
        _flatten_netlist_into(recnet, child_net, sep)
        for iname, iinstance in child_net["instances"].items():
            net["instances"][f"{name}{sep}{iname}"] = iinstance
        ports = {k: f"{name}{sep}{v}" for k, v in child_net["ports"].items()}
        net["connections"] = net.get("connections", {})
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
        child_net["connections"] = child_net.get("connections", {})
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
def convert_nets_to_connections(
    netlist: sax.Netlist,
) -> sax.Netlist: ...


@overload
def convert_nets_to_connections(
    netlist: sax.RecursiveNetlist,
) -> sax.RecursiveNetlist: ...


def convert_nets_to_connections(
    netlist: sax.AnyNetlist,
) -> sax.AnyNetlist:
    if (recnet := sax.try_into[sax.RecursiveNetlist](netlist)) is not None:
        return {k: convert_nets_to_connections(v) for k, v in recnet.items()}
    net: sax.Netlist = sax.into[sax.Netlist](netlist)
    nets = net.pop("nets", [])
    connections = net.get("connections", {})
    connections = _nets_to_connections(nets, connections)
    net["connections"] = connections
    return net


def _nets_to_connections(
    nets: sax.Nets, connections: sax.Connections
) -> sax.Connections:
    connections = dict(connections.items())
    inverse_connections = {v: k for k, v in connections.items()}

    def _is_connected(p: sax.InstancePort) -> bool:
        return (p in connections) or (p in inverse_connections)

    def _add_connection(p: sax.InstancePort, q: sax.InstancePort) -> None:
        connections[p] = q
        inverse_connections[q] = p

    def _get_connected_port(p: sax.InstancePort) -> sax.InstancePort:
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
            raise ValueError(
                msg,
            )
        if _is_connected(q):
            _p = _get_connected_port(q)
            msg = (
                "SAX currently does not support multiply connected ports. "
                f"Got {p}<->{q} and {_p}<->{q}"
            )
            raise ValueError(
                msg,
            )
        _add_connection(p, q)
    return connections
