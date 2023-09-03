""" SAX Circuit Definition """

from __future__ import annotations

import os
import shutil
import sys
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypedDict, Union

import black
import networkx as nx
import numpy as np

from .backends import circuit_backends
from .netlist import (
    Netlist,
    NetlistDict,
    RecursiveNetlist,
    RecursiveNetlistDict,
    remove_unused_instances,
)
from .saxtypes import Model, Settings, SType, scoo, sdense, sdict
from .utils import (
    _replace_kwargs,
    get_ports,
    get_settings,
    merge_dicts,
    update_settings,
)

try:
    from pydantic.v1 import ValidationError  # type: ignore
except ImportError:
    from pydantic import ValidationError  # type: ignore


class CircuitInfo(NamedTuple):
    """Information about the circuit function you created."""

    dag: nx.DiGraph
    models: Dict[str, Model]


def circuit(
    netlist: Union[Netlist, NetlistDict, RecursiveNetlist, RecursiveNetlistDict],
    models: Optional[Dict[str, Model]] = None,
    backend: str = "default",
    return_type: str = "sdict",
) -> Tuple[Model, CircuitInfo]:
    """create a circuit function for a given netlist"""
    netlist = _ensure_recursive_netlist_dict(netlist)

    # TODO: do the following two steps *after* recursive netlist parsing.
    netlist = remove_unused_instances(netlist)
    netlist, instance_models = _extract_instance_models(netlist)

    recnet: RecursiveNetlist = _validate_net(netlist)  # type: ignore
    dependency_dag: nx.DiGraph = _validate_dag(_create_dag(recnet, models))
    models = _validate_models({**(models or {}), **instance_models}, dependency_dag)
    backend = _validate_circuit_backend(backend)

    circuit = None
    new_models = {}
    current_models = {}
    model_names = list(nx.topological_sort(dependency_dag))[::-1]
    for model_name in model_names:
        if model_name in models:
            new_models[model_name] = models[model_name]
            continue

        flatnet = recnet.__root__[model_name]
        current_models.update(new_models)
        new_models = {}

        current_models[model_name] = circuit = _flat_circuit(
            flatnet.instances,
            flatnet.connections,
            flatnet.ports,
            current_models,
            backend,
        )

    assert circuit is not None
    circuit = _enforce_return_type(circuit, return_type)
    return circuit, CircuitInfo(dag=dependency_dag, models=current_models)


def _create_dag(
    netlist: RecursiveNetlist,
    models: Optional[Dict[str, Any]] = None,
):
    if models is None:
        models = {}
    assert isinstance(models, dict)

    all_models = {}
    g = nx.DiGraph()

    for model_name, subnetlist in netlist.dict()["__root__"].items():
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
    parent_node = next(iter(netlist.__root__.keys()))
    nodes = [parent_node, *nx.descendants(g, parent_node)]
    g = nx.induced_subgraph(g, nodes)

    return g


def _draw_dag(dag, with_labels=True, **kwargs):
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


def _patch_path():
    os_paths = {p: None for p in os.environ.get("PATH", "").split(os.pathsep)}
    sys_paths = {p: None for p in sys.path}
    other_paths = {os.path.dirname(sys.executable): None}
    os.environ["PATH"] = os.pathsep.join({**os_paths, **sys_paths, **other_paths})


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
    nodes = [n for n, d in g.in_degree() if d == 0]
    return nodes


def _find_leaves(g):
    nodes = [n for n, d in g.out_degree() if d == 0]
    return nodes


def _validate_models(models, dag):
    required_models = _find_leaves(dag)
    missing_models = [m for m in required_models if m not in models]
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
    return {**models}  # shallow copy


def _flat_circuit(instances, connections, ports, models, backend):
    analyze_fn, evaluate_fn = circuit_backends[backend]

    inst2model = {k: models[inst.component] for k, inst in instances.items()}
    inst_port_mode = {
        k: _port_modes_dict(get_ports(models[inst.component]))
        for k, inst in instances.items()
    }
    connections = _get_multimode_connections(connections, inst_port_mode)
    ports = _get_multimode_ports(ports, inst_port_mode)

    model_settings = {name: get_settings(model) for name, model in inst2model.items()}
    netlist_settings = {
        name: {
            k: v for k, v in (inst.settings or {}).items() if k in model_settings[name]
        }
        for name, inst in instances.items()
    }
    default_settings = merge_dicts(model_settings, netlist_settings)
    analyzed = analyze_fn(connections, ports)

    def _circuit(**settings: Settings) -> SType:
        full_settings = merge_dicts(default_settings, settings)
        full_settings = _forward_global_settings(inst2model, full_settings)

        instances: Dict[str, SType] = {}
        for inst_name, model in inst2model.items():
            instances[inst_name] = model(**full_settings.get(inst_name, {}))

        S = evaluate_fn(analyzed, instances)
        return S

    _replace_kwargs(_circuit, **default_settings)

    return _circuit


def _forward_global_settings(instances, settings):
    global_settings = {}
    for k in list(settings.keys()):
        if k in instances:
            continue
        global_settings[k] = settings.pop(k)
    if global_settings:
        settings = update_settings(settings, **global_settings)
    return settings


def _port_modes_dict(port_modes):
    result = {}
    for port_mode in port_modes:
        if "@" in port_mode:
            port, mode = port_mode.split("@")
        else:
            port, mode = port_mode, None
        if port not in result:
            result[port] = set()
        if mode is not None:
            result[port].add(mode)
    return result


def _get_multimode_connections(connections, inst_port_mode):
    mm_connections = {}
    for inst_port1, inst_port2 in connections.items():
        inst1, port1 = inst_port1.split(",")
        inst2, port2 = inst_port2.split(",")
        modes1 = inst_port_mode[inst1][port1]
        modes2 = inst_port_mode[inst2][port2]
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


def _get_multimode_ports(ports, inst_port_mode):
    mm_ports = {}
    for port, inst_port2 in ports.items():
        inst2, port2 = inst_port2.split(",")
        modes2 = inst_port_mode[inst2][port2]
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


def _ensure_recursive_netlist_dict(netlist):
    if not isinstance(netlist, dict):
        netlist = netlist.dict()
    if "__root__" in netlist:
        netlist = netlist["__root__"]
    if "instances" in netlist:
        netlist = {"top_level": netlist}
    netlist = {**netlist}
    for k, v in netlist.items():
        netlist[k] = {**v}
    return netlist


def _extract_instance_models(netlist):
    models = {}
    for netname, net in netlist.items():
        net = {**net}
        net["instances"] = {**net["instances"]}
        for name, inst in net["instances"].items():
            if callable(inst):
                settings = get_settings(inst)
                if isinstance(inst, partial) and inst.args:
                    raise ValueError(
                        "SAX circuits and netlists don't support partials "
                        "with positional arguments."
                    )
                while isinstance(inst, partial):
                    inst = inst.func
                models[inst.__name__] = inst
                net["instances"][name] = {
                    "component": inst.__name__,
                    "settings": settings,
                }
        netlist[netname] = net
    return netlist, models


def _validate_circuit_backend(backend):
    backend = backend.lower()
    # assert valid circuit_backend
    if backend not in circuit_backends:
        raise KeyError(
            f"circuit backend {backend} not found. Allowed circuit backends: "
            f"{', '.join(circuit_backends.keys())}."
        )
    return backend


def _validate_net(netlist: Union[Netlist, RecursiveNetlist]) -> RecursiveNetlist:
    if isinstance(netlist, dict):
        try:
            netlist = Netlist.parse_obj(netlist)
        except ValidationError:
            netlist = RecursiveNetlist.parse_obj(netlist)
    elif isinstance(netlist, Netlist):
        netlist = RecursiveNetlist(__root__={"top_level": netlist})
    return netlist


def _validate_dag(dag):
    nodes = _find_root(dag)
    if len(nodes) > 1:
        raise ValueError(f"Multiple top_levels found in netlist: {nodes}")
    if len(nodes) < 1:
        raise ValueError("Netlist does not contain any nodes.")
    if not dag.is_directed():
        raise ValueError("Netlist dependency cycles detected!")
    return dag


def get_required_circuit_models(
    netlist: Union[Netlist, NetlistDict, RecursiveNetlist, RecursiveNetlistDict],
    models: Optional[Dict[str, Model]] = None,
) -> List:
    """Figure out which models are needed for a given netlist"""
    if models is None:
        models = {}
    assert isinstance(models, dict)
    netlist = _ensure_recursive_netlist_dict(netlist)
    # TODO: do the following two steps *after* recursive netlist parsing.
    netlist = remove_unused_instances(netlist)
    netlist, _ = _extract_instance_models(netlist)
    recnet: RecursiveNetlist = _validate_net(netlist)  # type: ignore

    missing_models = {}
    missing_model_names = []
    g = nx.DiGraph()

    for model_name, subnetlist in recnet.dict()["__root__"].items():
        if model_name not in missing_models:
            missing_models[model_name] = models.get(model_name, subnetlist)
            g.add_node(model_name)
        if model_name in models:
            continue
        for instance in subnetlist["instances"].values():
            component = instance["component"]
            if (component not in missing_models) and (component not in models):
                missing_models[component] = models.get(component, None)
                missing_model_names.append(component)
                g.add_node(component)
            g.add_edge(model_name, component)
    return missing_model_names
