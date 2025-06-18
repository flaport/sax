"""SAX forward_only Backend."""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import networkx as nx

import sax

from ..s import scoo, sdict


def analyze_instances_forward(
    instances: sax.Instances,
    models: sax.Models,
) -> dict[str, sax.SCoo]:
    """Analyze instances for forward algorithm."""
    instances, instances_old = {}, instances
    for k, v in instances_old.items():
        instances[k] = sax.into[sax.Instance](v)
    model_names = set()
    for i in instances.values():
        model_names.add(i["component"])
    dummy_models = {k: scoo(models[k]()) for k in model_names}
    dummy_instances = {}
    for k, i in instances.items():
        dummy_instances[k] = dummy_models[i["component"]]
    return dummy_instances


def analyze_circuit_forward(
    analyzed_instances: dict[str, sax.SDict],  # noqa: ARG001
    connections: sax.Connections,
    ports: sax.Ports,
) -> Any:  # noqa: ANN401
    """Analyze circuit for forward algorithm."""
    return connections, ports


def evaluate_circuit_forward(
    analyzed: Any,  # noqa: ANN401
    instances: dict[sax.InstanceName, sax.SDict],
) -> sax.SDict:
    """Evaluate a circuit for the given sdicts using simple matrix multiplication."""
    connections, ports = analyzed
    edges = _graph_edges_directed(instances, connections, ports)

    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    # Dictionary to store signals at each node
    circuit_sdict = {}
    for in_port in ports:
        if in_port.startswith("in"):
            node_signals = {("", in_port): 1}
            bfs_output = nx.bfs_layers(graph, ("", in_port))
            for layer in bfs_output:
                layer_signals = {}
                for node in layer:
                    if node in node_signals:
                        signal = node_signals[node]
                        for neighbor in graph.successors(node):
                            transmission = cast(
                                int, graph[node][neighbor]["transmission"]
                            )
                            if neighbor in layer_signals:
                                layer_signals[neighbor] += signal * transmission
                            else:
                                layer_signals[neighbor] = signal * transmission
                node_signals.update(layer_signals)
            sdict = {
                (in_port, p2): v
                for (p1, p2), v in node_signals.items()
                if p1 == "" and p2.startswith("out")
            }
            circuit_sdict.update(sdict)
    return circuit_sdict


def _split_port(port: str) -> tuple[str, str]:
    try:
        instance, port = port.split(",")
    except ValueError:
        (port,) = port.split(",")
        instance = ""
    return instance, port


def _graph_edges_directed(
    instances: dict[str, sax.SDict],
    connections: sax.Connections,
    ports: sax.Ports,
) -> Any:  # noqa: ANN401
    one = jnp.array([1.0], dtype=float)
    edges_dict = {}
    edges_dict.update({_split_port(k): _split_port(v) for k, v in connections.items()})
    edges_dict.update({_split_port(k): _split_port(v) for k, v in ports.items()})
    edges = []
    for n1, n2 in edges_dict.items():
        if n1[0] == "" and n1[1].startswith("out"):
            edges += [(n2, n1, {"transmission": one})]
        else:
            edges += [(n1, n2, {"transmission": one})]

    for instance, s in instances.items():
        for (p1, p2), w in sdict(s).items():
            if p1.startswith("in") and p2.startswith("out"):
                edges += [
                    (
                        (instance, p1),
                        (instance, p2),
                        {"transmission": jnp.asarray(w, dtype=complex).ravel()},
                    )
                ]
    return edges
