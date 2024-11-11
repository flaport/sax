""" SAX Circuit Definition """

from __future__ import annotations

import os
import shutil
import sys
from functools import partial
from typing import Callable, NamedTuple

import black
import networkx as nx
import numpy as np

from .backends import circuit_backends, backend_map
from .netlist import AnyNetlist, Netlist, NetlistDict, RecursiveNetlist, is_recursive
from .netlist import netlist as parse_netlist
from .saxtypes import Model, Settings, SType, scoo, sdense, sdict
from .utils import (
    _replace_kwargs,
    get_ports,
    get_settings,
    merge_dicts,
    update_settings,
)


class CircuitInfo(NamedTuple):
    """Information about the circuit function you created."""

    dag: nx.DiGraph[str]
    models: dict[str, Model]
    backend: str


def circuit(
    netlist: AnyNetlist,
    models: dict[str, Model] | None = None,
    backend: str = "default",
    return_type: str = "sdict",
    ignore_missing_ports: bool = False,
) -> tuple[Model, CircuitInfo]:
    """Create a circuit function for a given netlist.

    Args:
        netlist: The netlist to create a circuit for.
        models: A dictionary of models to use in the circuit.
        backend: The backend to use for the circuit.
        return_type: The type of the circuit function to return.
        ignore_missing_ports: Ignore missing ports in the netlist.

    """
    backend = _validate_circuit_backend(backend)

    instance_models = _extract_instance_models(netlist)
    recnet: RecursiveNetlist = parse_netlist(
        netlist, with_unconnected_instances=False, with_placements=False
    )
    _validate_netlist_ports(recnet)
    dependency_dag: nx.DiGraph[str] = _create_dag(recnet, models, validate=True)
    models = _validate_models(models, dependency_dag, extra_models=instance_models)

    circuit = None
    new_models = {}
    current_models = {}
    model_names = list(nx.topological_sort(dependency_dag))[::-1]
    for model_name in model_names:
        if model_name in models:
            new_models[model_name] = models[model_name]
            continue

        flatnet = recnet.root[model_name]
        current_models |= new_models
        new_models = {}

        current_models[model_name] = circuit = _flat_circuit(
            flatnet.instances,
            flatnet.connections,
            flatnet.ports,
            current_models,
            backend,
            ignore_missing_ports=ignore_missing_ports,
        )

    assert circuit is not None
    circuit = _enforce_return_type(circuit, return_type)
    return circuit, CircuitInfo(
        dag=dependency_dag,
        models=current_models,
        backend=backend_map.get(backend, backend),
    )


def _create_dag(
    netlist: RecursiveNetlist,
    models: dict[str, Model] | None = None,
    validate: bool = False,
) -> nx.DiGraph[str]:
    if models is None:
        models = {}
    assert isinstance(models, dict)

    all_models = {}
    g = nx.DiGraph()

    for model_name, subnetlist in netlist.model_dump().items():
        if model_name not in all_models:
            all_models[model_name] = models.get(model_name, subnetlist)
            g.add_node(model_name)
        if model_name in models:
            continue
        for instance in subnetlist["instances"].values():
            component = instance["component"]
            if component not in all_models:
                all_models[component] = models.get(component, None)
                g.add_node(component)
            g.add_edge(model_name, component)

    # we only need the nodes that depend on the parent...
    parent_node = next(iter(netlist.root.keys()))
    nodes = [parent_node, *nx.descendants(g, parent_node)]
    g = nx.induced_subgraph(g, nodes)
    assert isinstance(g, nx.DiGraph)
    if validate:
        g = _validate_dag(g)
    return g


def draw_dag(dag, with_labels=True, **kwargs):
    _patch_path()
    if shutil.which("dot"):
        return nx.draw(
            dag,
            nx.nx_pydot.pydot_layout(dag, prog="dot"),
            with_labels=with_labels,
            **kwargs,
        )
    else:
        return nx.draw(dag, _my_dag_pos(dag), with_labels=with_labels, **kwargs)


def get_required_circuit_models(
    netlist: AnyNetlist,
    models: dict[str, Model] | None = None,
) -> list[str]:
    """Figure out which models are needed for a given netlist.

    Args:
        netlist: The netlist to create a circuit for.
        models: A dictionary of models to use in the circuit.

    """
    instance_models = _extract_instance_models(netlist)
    recnet: RecursiveNetlist = parse_netlist(
        netlist, with_unconnected_instances=False, with_placements=False
    )
    dependency_dag: nx.DiGraph[str] = _create_dag(recnet, models, validate=True)
    _, required, _ = _find_missing_models(
        models, dependency_dag, extra_models=instance_models
    )
    return required


def _flat_circuit(
    instances, connections, ports, models, backend, ignore_missing_ports=False
):
    analyze_insts_fn, analyze_fn, evaluate_fn = circuit_backends[backend]
    dummy_instances = analyze_insts_fn(instances, models)
    inst_port_mode = {
        k: _port_modes_dict(get_ports(s)) for k, s in dummy_instances.items()
    }
    connections = _get_multimode_connections(
        connections, inst_port_mode, ignore_missing_ports=ignore_missing_ports
    )
    ports = _get_multimode_ports(
        ports, inst_port_mode, ignore_missing_ports=ignore_missing_ports
    )

    inst2model = {}
    for k, inst in instances.items():
        inst2model[k] = models[inst.component]

    model_settings = {name: get_settings(model) for name, model in inst2model.items()}
    netlist_settings = {
        name: {
            k: v for k, v in (inst.settings or {}).items() if k in model_settings[name]
        }
        for name, inst in instances.items()
    }
    default_settings = merge_dicts(model_settings, netlist_settings)
    analyzed = analyze_fn(dummy_instances, connections, ports)

    def _circuit(**settings: Settings) -> SType:
        full_settings = merge_dicts(default_settings, settings)
        full_settings = _forward_global_settings(inst2model, full_settings)
        full_settings = merge_dicts(full_settings, settings)

        instances: dict[str, SType] = {}
        for inst_name, model in inst2model.items():
            instances[inst_name] = model(**full_settings.get(inst_name, {}))

        S = evaluate_fn(analyzed, instances)
        return S

    _replace_kwargs(_circuit, **default_settings)

    return _circuit


def _patch_path():
    os_paths = {p: None for p in os.environ.get("PATH", "").split(os.pathsep)}
    sys_paths = {p: None for p in sys.path}
    other_paths = {os.path.dirname(sys.executable): None}
    os.environ["PATH"] = os.pathsep.join(os_paths | sys_paths | other_paths)


def _my_dag_pos(dag):
    # inferior to pydot
    in_degree = {}
    for k, v in dag.in_degree():
        if v not in in_degree:
            in_degree[v] = []
        in_degree[v].append(k)

    widths = {k: len(vs) for k, vs in in_degree.items()}
    width = max(widths.values())

    horizontal_pos = {
        k: np.linspace(0, 1, w + 2)[1:-1] * width for k, w in widths.items()
    }

    pos = {}
    for k, vs in in_degree.items():
        for x, v in zip(horizontal_pos[k], vs):
            pos[v] = (x, -k)
    return pos


def _find_root(g):
    return [n for n, d in g.in_degree() if d == 0]


def _find_leaves(g):
    return [n for n, d in g.out_degree() if d == 0]


def _find_missing_models(
    models: dict | None, dag: nx.DiGraph, extra_models: dict | None = None
) -> tuple[dict[str, Callable], list[str], list[str]]:
    if extra_models is None:
        extra_models = {}
    if models is None:
        models = {}
    models = {**models, **extra_models}
    required_models = _find_leaves(dag)
    missing_models = [m for m in required_models if m not in models]
    return models, required_models, missing_models


def _validate_models(
    models: dict | None, dag: nx.DiGraph, extra_models: dict | None = None
) -> dict[str, Model]:
    models, required_models, missing_models = _find_missing_models(
        models, dag, extra_models
    )
    if missing_models:
        model_diff = {
            "Missing Models": missing_models,
            "Given Models": list(models),
            "Required Models": required_models,
        }
        raise ValueError(
            "Missing models. The following models are still missing to build "
            f"the circuit:\n{black.format_str(repr(model_diff), mode=black.Mode())}"
        )
    return models


def _forward_global_settings(instances, settings):
    global_settings = {
        k: settings.pop(k) for k in list(settings.keys()) if k not in instances
    }
    if global_settings:
        settings = update_settings(settings, **global_settings)
    return settings


def _port_modes_dict(port_modes):
    result = {}
    for port_mode in port_modes:
        port, mode = port_mode.split("@") if "@" in port_mode else (port_mode, None)
        if port not in result:
            result[port] = set()
        if mode is not None:
            result[port].add(mode)
    return result


def _get_multimode_connections(connections, inst_port_mode, ignore_missing_ports=False):
    mm_connections = {}
    for inst_port1, inst_port2 in connections.items():
        inst1, port1 = inst_port1.split(",")
        inst2, port2 = inst_port2.split(",")
        try:
            modes1 = inst_port_mode[inst1][port1]
        except KeyError:
            if ignore_missing_ports:
                continue
            raise RuntimeError(
                f"Instance {inst1} does not contain port {port1}. Available ports: {list(inst_port_mode[inst1])}."
            )
        try:
            modes2 = inst_port_mode[inst2][port2]
        except KeyError:
            if ignore_missing_ports:
                continue
            raise RuntimeError(
                f"Instance {inst2} does not contain port {port2}. Available ports: {list(inst_port_mode[inst2])}."
            )
        if not modes1 and not modes2:
            mm_connections[f"{inst1},{port1}"] = f"{inst2},{port2}"
        elif (not modes1) or (not modes2):
            raise ValueError(
                "trying to connect a multimode model to single mode model.\n"
                "Please update your models dictionary.\n"
                f"Problematic connection: '{inst_port1}':'{inst_port2}'"
            )
        else:
            common_modes = modes1.intersection(modes2)
            for mode in sorted(common_modes):
                mm_connections[f"{inst1},{port1}@{mode}"] = f"{inst2},{port2}@{mode}"
    return mm_connections


def _get_multimode_ports(ports, inst_port_mode, ignore_missing_ports=False):
    mm_ports = {}
    for port, inst_port2 in ports.items():
        inst2, port2 = inst_port2.split(",")
        try:
            modes2 = inst_port_mode[inst2][port2]
        except KeyError:
            if ignore_missing_ports:
                continue
            raise RuntimeError(
                f"Instance {inst2} does not contain port {port2}. Available ports: {list(inst_port_mode[inst2])}"
            )
        if not modes2:
            mm_ports[port] = f"{inst2},{port2}"
        else:
            for mode in sorted(modes2):
                mm_ports[f"{port}@{mode}"] = f"{inst2},{port2}@{mode}"
    return mm_ports


def _enforce_return_type(model, return_type):
    stype_func = {
        "default": lambda x: x,
        "stype": lambda x: x,
        "sdict": sdict,
        "scoo": scoo,
        "sdense": sdense,
    }[return_type]
    return stype_func(model)


def _extract_instance_models(netlist: AnyNetlist) -> dict[str, Model]:
    if isinstance(netlist, Netlist):
        return {}
    elif isinstance(netlist, RecursiveNetlist):
        return {}
    elif isinstance(netlist, dict):
        if is_recursive(netlist):
            models = {}
            for net in netlist.values():
                models.update(_extract_instance_models(net))  # type: ignore
            return models
        else:
            callable_instances = [
                f for f in netlist["instances"].values() if callable(f)
            ]
            models = {}
            for f in callable_instances:
                while isinstance(f, partial):
                    f = f.func
                models[f.__name__] = f
            return models
    else:
        return {}


def _validate_circuit_backend(backend):
    backend = backend.lower()
    # assert valid circuit_backend
    if backend not in circuit_backends:
        raise KeyError(
            f"circuit backend {backend} not found. Allowed circuit backends: "
            f"{', '.join(circuit_backends.keys())}."
        )
    return backend


def _validate_dag(dag):
    nodes = _find_root(dag)
    if len(nodes) > 1:
        raise ValueError(f"Multiple top_levels found in netlist: {nodes}")
    if len(nodes) < 1:
        raise ValueError("Netlist does not contain any nodes.")
    if not dag.is_directed():
        raise ValueError("Netlist dependency cycles detected!")
    return dag


def _validate_netlist_ports(netlist: RecursiveNetlist):
    if len(netlist.root) < 1:
        raise ValueError("Cannot create circuit: empty netlist")
    net: Netlist = netlist.root[list(netlist.root)[0]]
    ports_str = ", ".join(list(net.ports))
    if not ports_str:
        ports_str = "no ports given"
    if len(net.ports) < 2:
        raise ValueError(
            "Cannot create circuit: "
            f"at least 2 ports need to be defined. Got {ports_str}."
        )
