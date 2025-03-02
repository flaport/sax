"""SAX Circuit Definition."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any, Literal, cast, overload

import networkx as nx
import numpy as np

import sax

from .backends import circuit_backends, validate_circuit_backend
from .netlist import convert_nets_to_connections, remove_unused_instances
from .netlist import netlist as into_recnet
from .s import get_ports, scoo, sdense, sdict
from .utils import get_settings, merge_dicts, replace_kwargs, update_settings

__all__ = ["circuit"]


@overload
def circuit(
    netlist: sax.AnyNetlist,
    models: sax.Models | None = None,
    backend: sax.Backend | Literal["default"] = "default",
    return_type: Literal["SDict"] = "SDict",
    *,
    top_level_name: str = "top_level",
    ignore_impossible_connections: bool = False,
) -> tuple[sax.SDictModel, sax.CircuitInfo]: ...


@overload
def circuit(
    netlist: sax.AnyNetlist,
    models: sax.Models | None = None,
    backend: sax.Backend | Literal["default"] = "default",
    return_type: Literal["SDense"] = "SDense",
    *,
    top_level_name: str = "top_level",
    ignore_impossible_connections: bool = False,
) -> tuple[sax.SDenseModel, sax.CircuitInfo]: ...


@overload
def circuit(
    netlist: sax.AnyNetlist,
    models: sax.Models | None = None,
    backend: sax.Backend | Literal["default"] = "default",
    return_type: Literal["SCoo"] = "SCoo",
    *,
    top_level_name: str = "top_level",
    ignore_impossible_connections: bool = False,
) -> tuple[sax.SCooModel, sax.CircuitInfo]: ...


def circuit(  # noqa: PLR0913
    netlist: sax.AnyNetlist,
    models: sax.Models | None = None,
    backend: sax.Backend | Literal["default"] = "default",
    return_type: Literal["SDict", "SDense", "SCoo"] = "SDense",
    *,
    top_level_name: str = "top_level",
    ignore_impossible_connections: bool = False,
) -> tuple[sax.Model, sax.CircuitInfo]:
    """Create a circuit function for a given netlist.

    Args:
        netlist: The netlist to create a circuit for.
        models: A dictionary of models to use in the circuit.
        backend: The backend to use for the circuit.
        return_type: The type of the circuit function to return.
        top_level_name: select a circuit from the recnet to be considered 'top level'.
            Ignored when no matching name found in recursive netlist.
        ignore_impossible_connections: Ignore connections to missing instance ports
            in stead of erroring out.

    """
    backend = validate_circuit_backend(backend)

    instance_models = _extract_instance_models(netlist)
    recnet = into_recnet(
        netlist,
        top_level_name=top_level_name,
    )
    recnet = convert_nets_to_connections(recnet)
    recnet = remove_unused_instances(recnet)
    _validate_netlist_ports(recnet)
    dependency_dag = _create_dag(recnet, models, validate=True)
    models = _validate_models(
        models or {}, dependency_dag, extra_models=instance_models
    )

    circuit = None
    new_models = {}
    current_models = {}
    model_names = list(nx.topological_sort(dependency_dag))[::-1]
    for model_name in model_names:
        if model_name in models:
            new_models[model_name] = models[model_name]
            continue

        flatnet = recnet[model_name]
        current_models |= new_models
        new_models = {}

        current_models[model_name] = circuit = _flat_circuit(
            flatnet["instances"],
            flatnet.get("connections", {}),
            flatnet["ports"],
            current_models,
            backend,
            ignore_impossible_connections=ignore_impossible_connections,
        )

    if circuit is None:
        msg = "Could not construct circuit (unknown reason)"
        raise RuntimeError(msg)
    circuit = _enforce_return_type(circuit, return_type)
    return circuit, sax.CircuitInfo(
        dag=dependency_dag,
        models=current_models,
        backend=backend,
    )


def _create_dag(
    netlist: sax.RecursiveNetlist,
    models: sax.Models | None = None,
    *,
    validate: bool = False,
) -> nx.DiGraph[Any]:
    if models is None:
        models = {}

    all_models = {}
    g = nx.DiGraph()

    for model_name, subnetlist in netlist.items():
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
    parent_node = next(iter(netlist.keys()))
    nodes = [parent_node, *nx.descendants(g, parent_node)]
    g = cast(nx.DiGraph[Any], nx.induced_subgraph(g, nodes))
    if validate:
        g = _validate_dag(g)
    return g


def draw_dag(dag: nx.DiGraph[Any], *, with_labels: bool = True, **kwargs) -> None:
    if shutil.which("dot"):
        return nx.draw(
            dag,
            nx.nx_pydot.pydot_layout(dag, prog="dot"),
            with_labels=with_labels,
            **kwargs,
        )
    return nx.draw(dag, _my_dag_pos(dag), with_labels=with_labels, **kwargs)


def get_required_circuit_models(
    netlist: sax.AnyNetlist,
    models: dict[str, sax.Model] | None = None,
) -> list[str]:
    """Figure out which models are needed for a given netlist.

    Args:
        netlist: The netlist to create a circuit for.
        models: A dictionary of models to use in the circuit.

    """
    instance_models = _extract_instance_models(netlist)
    recnet: sax.RecursiveNetlist = into_recnet(
        netlist,
    )
    recnet = remove_unused_instances(recnet)
    dependency_dag = _create_dag(recnet, models, validate=True)
    _, required, _ = _find_missing_models(
        models,
        dependency_dag,
        extra_models=instance_models,
    )
    return required


def _flat_circuit(  # noqa: PLR0913
    instances: sax.Instances,
    connections: sax.Connections,
    ports: sax.Ports,
    models: sax.Models,
    backend: sax.Backend,
    *,
    ignore_impossible_connections: bool = False,
) -> sax.Model:
    analyze_insts_fn, analyze_fn, evaluate_fn = circuit_backends[backend]
    dummy_instances = analyze_insts_fn(instances, models)
    inst_port_mode = {
        k: _port_modes_dict(get_ports(s)) for k, s in dummy_instances.items()
    }
    connections = _get_multimode_connections(
        connections,
        inst_port_mode,
        ignore_impossible_connections=ignore_impossible_connections,
    )
    ports = _get_multimode_ports(
        ports,
        inst_port_mode,
        ignore_impossible_connections=ignore_impossible_connections,
    )

    inst2model = {}
    for k, inst in instances.items():
        inst2model[k] = models[inst["component"]]

    model_settings = {name: get_settings(model) for name, model in inst2model.items()}
    netlist_settings = {
        name: {
            k: v
            for k, v in (inst.get("settings") or {}).items()
            if k in model_settings[name]
        }
        for name, inst in instances.items()
    }
    default_settings = merge_dicts(model_settings, netlist_settings)
    analyzed = analyze_fn(dummy_instances, connections, ports)

    def _circuit(**settings: sax.SettingsValue) -> sax.SType:
        full_settings = merge_dicts(default_settings, settings)
        full_settings = _forward_global_settings(inst2model, full_settings)
        full_settings = merge_dicts(full_settings, settings)

        instances: dict[str, sax.SType] = {}
        for inst_name, model in inst2model.items():
            instances[inst_name] = model(**full_settings.get(inst_name, {}))

        return evaluate_fn(analyzed, instances)

    replace_kwargs(_circuit, **default_settings)

    return cast(sax.Model, _circuit)


def _in_degree(dag: nx.DiGraph[Any]) -> Iterator[tuple[str, int]]:
    return cast(Iterator[tuple[str, int]], dag.in_degree())


def _out_degree(dag: nx.DiGraph[Any]) -> Iterator[tuple[str, int]]:
    return cast(Iterator[tuple[str, int]], dag.out_degree())


def _my_dag_pos(dag: nx.DiGraph[Any]) -> dict:
    # inferior to pydot
    in_degree = {}
    for k, v in _in_degree(dag):
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
        for x, v in zip(horizontal_pos[k], vs, strict=False):
            pos[v] = (x, -k)
    return pos


def _find_root(g: nx.DiGraph[Any]) -> list[str]:
    return [n for n, d in _in_degree(g) if d == 0]


def _find_leaves(g: nx.DiGraph[Any]) -> list[str]:
    return [n for n, d in _out_degree(g) if d == 0]


def _find_missing_models(
    models: dict | None,
    dag: nx.DiGraph[Any],
    extra_models: dict | None = None,
) -> tuple[sax.Models, list[str], list[str]]:
    if extra_models is None:
        extra_models = {}
    if models is None:
        models = {}
    models = {**models, **extra_models}
    required_models = _find_leaves(dag)
    missing_models = [m for m in required_models if m not in models]
    return models, required_models, missing_models


def _validate_models(
    models: sax.Models,
    dag: nx.DiGraph[Any],
    extra_models: dict | None = None,
) -> sax.Models:
    models, required_models, missing_models = _find_missing_models(
        models,
        dag,
        extra_models,
    )
    if missing_models:
        model_diff = {
            "Missing Models": missing_models,
            "Given Models": list(models),
            "Required Models": required_models,
        }
        model_diff_str = json.dumps(model_diff, indent=4)
        msg = (
            "Missing models. The following models are still missing to build "
            f"the circuit:\n{model_diff_str}"
        )
        raise ValueError(msg)
    return models


def _forward_global_settings(
    instances: sax.Instances, settings: sax.Settings
) -> sax.Settings:
    global_settings = {
        k: settings.pop(k) for k in list(settings.keys()) if k not in instances
    }
    if global_settings:
        settings = update_settings(settings, **global_settings)
    return settings


def _port_modes_dict(
    port_modes: Iterable[sax.PortMode],
) -> dict[sax.Port, set[sax.Mode]]:
    result = {}
    for port_mode in port_modes:
        port, mode = port_mode.split("@") if "@" in port_mode else (port_mode, None)
        if port not in result:
            result[port] = set()
        if mode is not None:
            result[port].add(mode)
    return result


def _get_multimode_connections(
    connections: sax.Connections,
    inst_port_mode: dict[sax.InstanceName, dict[sax.Port, set[sax.Mode]]],
    *,
    ignore_impossible_connections: bool = False,
) -> sax.Connections:
    mm_connections = {}
    for inst_port1, inst_port2 in connections.items():
        inst1, port1 = inst_port1.split(",")
        inst2, port2 = inst_port2.split(",")
        try:
            modes1 = inst_port_mode[inst1][port1]
        except KeyError as e:
            if ignore_impossible_connections:
                continue
            msg = (
                f"Instance {inst1} does not contain port {port1}. "
                f"Available ports: {list(inst_port_mode[inst1])}."
            )
            raise KeyError(msg) from e
        try:
            modes2 = inst_port_mode[inst2][port2]
        except KeyError as e:
            if ignore_impossible_connections:
                continue
            msg = (
                f"Instance {inst2} does not contain port {port2}. "
                f"Available ports: {list(inst_port_mode[inst2])}."
            )
            raise KeyError(msg) from e
        if not modes1 and not modes2:
            mm_connections[f"{inst1},{port1}"] = f"{inst2},{port2}"
        elif (not modes1) or (not modes2):
            msg = (
                "trying to connect a multimode model to single mode model.\n"
                "Please update your models dictionary.\n"
                f"Problematic connection: '{inst_port1}':'{inst_port2}'"
            )
            raise ValueError(
                msg,
            )
        else:
            common_modes = modes1.intersection(modes2)
            for mode in sorted(common_modes):
                mm_connections[f"{inst1},{port1}@{mode}"] = f"{inst2},{port2}@{mode}"
    return mm_connections


def _get_multimode_ports(
    ports: sax.Ports,
    inst_port_mode: dict[sax.InstanceName, dict[sax.Port, set[sax.Mode]]],
    *,
    ignore_impossible_connections: bool = False,
) -> sax.Ports:
    mm_ports = {}
    for port, inst_port2 in ports.items():
        inst2, port2 = inst_port2.split(",")
        try:
            modes2 = inst_port_mode[inst2][port2]
        except KeyError as e:
            if ignore_impossible_connections:
                continue
            msg = (
                f"Instance {inst2} does not contain port {port2}. "
                f"Available ports: {list(inst_port_mode[inst2])}"
            )
            raise KeyError(msg) from e
        if not modes2:
            mm_ports[port] = f"{inst2},{port2}"
        else:
            for mode in sorted(modes2):
                mm_ports[f"{port}@{mode}"] = f"{inst2},{port2}@{mode}"
    return mm_ports


def _enforce_return_type(model: sax.Model, return_type: Any) -> sax.Model:  # noqa: ANN401
    stypes = {
        "sdict": sdict,
        "scoo": scoo,
        "sdense": sdense,
        sax.SDict: sdict,
        sax.SDense: sdense,
        sax.SCoo: scoo,
        sax.SDictModel: sdict,
        sax.SDenseModel: sdense,
        sax.SCooModel: scoo,
    }
    if isinstance(return_type, str):
        return_type = return_type.lower()
    stype = stypes.get(return_type)
    if stype is None:
        return model
    return stype(model)


def _extract_instance_models(netlist: sax.AnyNetlist) -> sax.Models:
    if (recnet := sax.try_into[sax.RecursiveNetlist](netlist)) is not None:
        models = {}
        for net in recnet.values():
            models.update(_extract_instance_models(net))
        return models
    if (net := sax.try_into[sax.Netlist](netlist)) is not None:
        callable_instances = [f for f in net["instances"].values() if callable(f)]
        models = {}
        for f in callable_instances:
            while isinstance(f, partial):
                f = f.func
            models[f.__name__] = f
        return models
    return {}


def _validate_dag(dag: nx.DiGraph[Any]) -> nx.DiGraph[Any]:
    nodes = _find_root(dag)
    if len(nodes) > 1:
        msg = f"Multiple top_levels found in netlist: {nodes}"
        raise ValueError(msg)
    if len(nodes) < 1:
        msg = "Netlist does not contain any nodes."
        raise ValueError(msg)
    if not dag.is_directed():
        msg = "Netlist dependency cycles detected!"
        raise ValueError(msg)
    return dag


def _validate_netlist_ports(netlist: sax.RecursiveNetlist) -> None:
    top_level_name = next(iter(netlist))
    top_level = netlist[top_level_name]
    ports_str = ", ".join(list(top_level["ports"]))
    if not ports_str:
        ports_str = "no ports given"
    if len(top_level["ports"]) < 2:  # noqa: PLR2004
        msg = (
            "Cannot create circuit: "
            f"at least 2 ports need to be defined. Got {ports_str}."
        )
        raise ValueError(msg)
