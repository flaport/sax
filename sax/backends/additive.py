""" SAX Additive Backend """

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax.numpy as jnp
import networkx as nx

from ..netlist import Component
from ..saxtypes import Model, SDict, sdict


def analyze_instances_additive(
    instances: Dict[str, Component],
    models: Dict[str, Model],
) -> Dict[str, SDict]:
    instances, instances_old = {}, instances
    for k, v in instances_old.items():
        if not isinstance(v, Component):
            v = Component(**v)
        instances[k] = v
    model_names = set()
    for i in instances.values():
        model_names.add(i.component)
    dummy_models = {k: sdict(models[k]()) for k in model_names}
    dummy_instances = {}
    for k, i in instances.items():
        dummy_instances[k] = dummy_models[i.component]
    return dummy_instances


def analyze_circuit_additive(
    analyzed_instances: Dict[str, SDict],
    connections: Dict[str, str],
    ports: Dict[str, str],
) -> Any:
    return connections, ports


def evaluate_circuit_additive(
    analyzed: Any,
    instances: Dict[str, SDict],
) -> SDict:
    """evaluate a circuit for the given sdicts."""
    connections, ports = analyzed
    edges = _graph_edges(instances, connections, ports)

    graph = nx.Graph()
    graph.add_edges_from(edges)
    _prune_internal_output_nodes(graph)

    sdict = {}
    for source in ports:
        for target in ports:
            paths = _get_possible_paths(graph, source=("", source), target=("", target))
            if not paths:
                continue
            sdict[source, target] = _path_lengths(graph, paths)

    return sdict


def _split_port(port: str) -> Tuple[str, str]:
    try:
        instance, port = port.split(",")
    except ValueError:
        (port,) = port.split(",")
        instance = ""
    return instance, port


def _graph_edges(
    instances: Dict[str, SDict],
    connections: Dict[str, str],
    ports: Dict[str, str],
):
    zero = jnp.array([0.0], dtype=float)
    edges = {}
    edges.update({_split_port(k): _split_port(v) for k, v in connections.items()})
    edges.update({_split_port(v): _split_port(k) for k, v in connections.items()})
    edges.update({_split_port(k): _split_port(v) for k, v in ports.items()})
    edges.update({_split_port(v): _split_port(k) for k, v in ports.items()})
    edges = [(n1, n2, {"type": "C", "length": zero}) for n1, n2 in edges.items()]

    _instances = {
        **{i1: None for (i1, _), (_, _), _ in edges},
        **{i2: None for (_, _), (i2, _), _ in edges},
    }

    if "" in _instances:
        del _instances[""]  # external ports don't belong to an instance

    for instance in _instances:
        s = instances[instance]
        edges += [
            (
                (instance, p1),
                (instance, p2),
                {"type": "S", "length": jnp.asarray(length, dtype=float).ravel()},
            )
            for (p1, p2), length in sdict(s).items()
        ]

    return edges


def _prune_internal_output_nodes(graph):
    broken = True
    while broken:
        broken = False
        for (i, p), dic in list(graph.adjacency()):
            if (
                i != ""
                and len(dic) == 2
                and all(prop.get("type", "C") == "C" for prop in dic.values())
            ):
                graph.remove_node((i, p))
                graph.add_edge(*dic.keys(), type="C", length=0.0)
                broken = True
                break
    return graph


def _get_possible_paths(graph, source, target):
    paths = []
    default_props = {"type": "C", "length": 0.0}
    for path in nx.all_simple_edge_paths(graph, source, target):
        prevtype = "C"
        for n1, n2 in path:
            curtype = graph.get_edge_data(n1, n2, default_props)["type"]
            if curtype == prevtype == "S":
                break
            else:
                prevtype = curtype
        else:
            paths.append(path)
    return paths


def _path_lengths(graph, paths):
    lengths = []
    for path in paths:
        length = zero = jnp.array([0.0], dtype=float)
        default_edge_data = {"type": "C", "length": zero}
        for edge in path:
            edge_data = graph.get_edge_data(*edge, default_edge_data)
            length = (length[None, :] + edge_data.get("length", zero)[:, None]).ravel()
        lengths.append(length)
    return lengths
