# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: sax
#     language: python
#     name: sax
# ---

# +
# default_exp circuit
# -

# # Circuit
#
# > SAX Circuits

# Let's start where we left off (see [Netlist](06_netlist.ipynb)).

# hide
import os
os.environ["LOGURU_LEVEL"] = "CRITICAL"
import warnings
warnings.filterwarnings("ignore")

# +
# export
from __future__ import annotations

import os
import shutil
import sys
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TypedDict, Union

import black
import networkx as nx
import numpy as np
from pydantic import ValidationError
from sax import reciprocal
from sax.backends import circuit_backends
from sax.multimode import multimode, singlemode
from sax.netlist import Netlist, RecursiveNetlist
from sax.netlist_cleaning import remove_unused_instances
from sax.typing_ import Model, Settings, SType
from sax.utils import _replace_kwargs, get_settings, merge_dicts, update_settings
# -

# Let's start by creating a simple recursive netlist with gdsfactory.
#
# :::{note}
# We are using gdsfactory to create our netlist because it allows us to see the circuit we want to simulate and because we're striving to have a compatible netlist implementation in SAX.
#
# However... gdsfactory is not a dependency of SAX. You can also define your circuits by hand (see [SAX Quick Start](../examples/01_quick_start.ipynb) or you can use another tool to programmatically construct your netlists.
# :::

# +
import gdsfactory as gf
from IPython.display import display
from gdsfactory.components import mzi
from gdsfactory.get_netlist import get_netlist_recursive, get_netlist, get_netlist_yaml


@gf.cell
def twomzi():
    c = gf.Component()

    # instances
    mzi1 = mzi(delta_length=10)
    mzi2 = mzi(delta_length=20)

    # references
    mzi1_ = c << mzi1
    mzi2_ = c << mzi2

    # connections
    mzi2_.connect("o1", mzi1_.ports["o2"])

    # ports
    c.add_port("o1", port=mzi1_.ports["o1"])
    c.add_port("o2", port=mzi2_.ports["o2"])
    return c


comp = twomzi()
display(comp)
recnet = RecursiveNetlist.parse_obj(
    get_netlist_recursive(
        comp, get_netlist_func=partial(get_netlist, full_settings=True)
    )
)
flatnet = recnet.__root__["mzi_delta_length10"]


# -

# To be able to model this device we'll need some SAX dummy models:

# hide
def bend_euler(
    angle=90.0,
    p=0.5,
    # cross_section="strip",
    # direction="ccw",
    # with_bbox=True,
    # with_arc_floorplan=True,
    # npoints=720,
):
    return reciprocal({
        ('o1', 'o2'): 1.0
    })


# hide
def mmi1x2(
    width=0.5,
    width_taper= 1.0,
    length_taper= 10.0,
    length_mmi= 5.5,
    width_mmi= 2.5,
    gap_mmi= 0.25,
    # cross_section= strip,
    # taper= {function= taper},
    # with_bbox= True,
):
    return reciprocal({
        ('o1', 'o2'): 0.45**0.5,
        ('o1', 'o3'): 0.45**0.5,
    })


# hide
def mmi2x2(
    width=0.5,
    width_taper= 1.0,
    length_taper= 10.0,
    length_mmi= 5.5,
    width_mmi= 2.5,
    gap_mmi= 0.25,
    # cross_section= strip,
    # taper= {function= taper},
    # with_bbox= True,
):
    return reciprocal({
        ('o1', 'o3'): 0.45**0.5,
        ('o1', 'o4'): 1j * 0.45**0.5,
        ('o2', 'o3'): 1j * 0.45**0.5,
        ('o2', 'o4'): 0.45**0.5,
    })


# hide
def straight(
    length=0.01,
    #npoints=2,
    #with_bbox=True,
    #cross_section=...
):
    return reciprocal({
        ('o1', 'o2'): 1.0
    })


# In SAX, we usually aggregate the available models in a models dictionary:

models = {
    'straight': straight,
    'bend_euler': bend_euler,
    'mmi1x2': mmi1x2,
}


# We can now represent our recursive netlist model as a Directed Acyclic Graph:

# export
def create_dag(
    netlist: RecursiveNetlist,
    models: Optional[Dict[str, Any]] = None,
):
    if models is None:
        models = {}
    assert isinstance(models, dict)

    all_models = {}
    g = nx.DiGraph()

    for model_name, subnetlist in netlist.dict()["__root__"].items():
        if not model_name in all_models:
            all_models[model_name] = models.get(model_name, subnetlist)
            g.add_node(model_name)
        if model_name in models:
            continue
        for instance in subnetlist["instances"].values():
            component = instance["component"]
            if not component in all_models:
                all_models[component] = models.get(component, None)
                g.add_node(component)
            g.add_edge(model_name, component)

    # we only need the nodes that depend on the parent...
    parent_node = next(iter(netlist.__root__.keys()))
    nodes = [parent_node, *nx.descendants(g, parent_node)]
    g = nx.induced_subgraph(g, nodes)

    return g

# +
# export


def draw_dag(dag, with_labels=True, **kwargs):
    _patch_path()
    if shutil.which("dot"):
        return nx.draw(
            dag,
            nx.nx_pydot.pydot_layout(dag, prog="dot"),
            with_labels=with_labels,
            **kwargs
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
    height = max(widths) + 1

    horizontal_pos = {
        k: np.linspace(0, 1, w + 2)[1:-1] * width for k, w in widths.items()
    }

    pos = {}
    for k, vs in in_degree.items():
        for x, v in zip(horizontal_pos[k], vs):
            pos[v] = (x, -k)
    return pos


# -

dag = create_dag(recnet, models)
draw_dag(dag)

# Note that the DAG depends on the models we supply. We could for example stub one of the sub-netlists by a pre-defined model:

dag_ = create_dag(recnet, {**models, 'mzi_delta_length10': mmi2x2})
draw_dag(dag_, with_labels=True)


# This is useful if we for example pre-calculated a certain model.

# We can easily find the root of the DAG:

# export
def find_root(g):
    nodes = [n for n, d in g.in_degree() if d == 0]
    return nodes


find_root(dag)


# Similarly we can find the leaves:

# export
def find_leaves(g):
    nodes = [n for n, d in g.out_degree() if d == 0]
    return nodes


find_leaves(dag)


# To be able to simulate the circuit, we need to supply a model for each of the leaves in the dependency DAG. Let's write a validator that checks this

# export
def _validate_models(models, dag):
    required_models = find_leaves(dag)
    missing_models = [m for m in required_models if m not in models]
    if missing_models:
        model_diff = {
            "Missing Models": missing_models,
            "Given Models": list(models),
            "Required Models": required_models,
        }
        raise ValueError(
            "Missing models. The following models are still missing to build the circuit:\n"
            f"{black.format_str(repr(model_diff), mode=black.Mode())}"
        )
    return {**models} # shallow copy


models = _validate_models(models, dag)


# We can now dow a bottom-up simulation. Since at the bottom of the DAG, our circuit is always flat (i.e. not hierarchical) we can implement a minimal `_flat_circuit` definition, which only needs to work on a flat (non-hierarchical circuit):

# +
# export
def _flat_circuit(instances, connections, ports, models, backend):
    evaluate_circuit = circuit_backends[backend]

    inst2model = {k: models[inst.component] for k, inst in instances.items()}

    model_settings = {name: get_settings(model) for name, model in inst2model.items()}
    netlist_settings = {
        name: {k: v for k, v in (inst.settings or {}).items() if k in model_settings[name]}
        for name, inst in instances.items()
    }
    default_settings = merge_dicts(model_settings, netlist_settings)

    def _circuit(**settings: Settings) -> SType:
        settings = merge_dicts(default_settings, settings)
        settings = _forward_global_settings(inst2model, settings)

        instances: Dict[str, SType] = {}
        for inst_name, model in inst2model.items():
            instances[inst_name] = model(**settings.get(inst_name, {}))
        #print(f"{instances=}")
        #print(f"{connections=}")
        #print(f"{ports=}")
        S = evaluate_circuit(instances, connections, ports)
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


# -

flatnet = recnet.__root__['mzi_delta_length10']
single_mzi = _flat_circuit(flatnet.instances, flatnet.connections, flatnet.ports, models, "default")

# The resulting circuit is just another SAX model (i.e. a python function) returing an SType:

# +
# single_mzi?
# -

# Let's 'execute' the circuit:

single_mzi()


# Now that we can handle flat circuits the extension to hierarchical circuits is not so difficult:

# +
# export

def circuit(
    netlist: Union[Netlist, NetlistDict, RecursiveNetlist, RecursiveNetlistDict],
    models: Optional[Dict[str, Model]] = None,
    modes: Optional[List[str]] = None,
    backend: str = "default",
) -> Tuple[Model, CircuitInfo]:
    netlist = _ensure_recursive_netlist_dict(netlist)

    # TODO: do the following two steps *after* recursive netlist parsing.
    netlist = remove_unused_instances(netlist)
    netlist, instance_models = _extract_instance_models(netlist)

    recnet: RecursiveNetlist = _validate_net(netlist)
    dependency_dag: nx.DiGraph = _validate_dag(
        create_dag(recnet, models)
    )  # directed acyclic graph
    models = _validate_models({**(models or {}), **instance_models}, dependency_dag)
    modes = _validate_modes(modes)
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

        connections, ports, new_models = _make_singlemode_or_multimode(
            flatnet, modes, new_models
        )
        current_models.update(new_models)
        new_models = {}

        current_models[model_name] = circuit = _flat_circuit(
            flatnet.instances, connections, ports, current_models, backend
        )

    assert circuit is not None
    return circuit, CircuitInfo(dag=dependency_dag, models=current_models)


class NetlistDict(TypedDict):
    instances: Dict
    connections: Dict[str, str]
    ports: Dict[str, str]


RecursiveNetlistDict = Dict[str, NetlistDict]


class CircuitInfo(NamedTuple):
    dag: nx.DiGraph
    models: Dict[str, Model]


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
                        "SAX circuits and netlists don't support partials with positional arguments."
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


def _validate_modes(modes) -> List[str]:
    if modes is None:
        return ["te"]
    elif not modes:
        return ["te"]
    elif isinstance(modes, str):
        return [modes]
    elif all(isinstance(m, str) for m in modes):
        return modes
    else:
        raise ValueError(f"Invalid modes given: {modes}")


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
    nodes = find_root(dag)
    if len(nodes) > 1:
        raise ValueError(f"Multiple top_levels found in netlist: {nodes}")
    if len(nodes) < 1:
        raise ValueError(f"Netlist does not contain any nodes.")
    if not dag.is_directed():
        raise ValueError("Netlist dependency cycles detected!")
    return dag


def _make_singlemode_or_multimode(netlist, modes, models):
    if len(modes) == 1:
        connections, ports, models = _make_singlemode(netlist, modes[0], models)
    else:
        connections, ports, models = _make_multimode(netlist, modes, models)
    return connections, ports, models


def _make_singlemode(netlist, mode, models):
    models = {k: singlemode(m, mode=mode) for k, m in models.items()}
    return netlist.connections, netlist.ports, models


def _make_multimode(netlist, modes, models):
    models = {k: multimode(m, modes=modes) for k, m in models.items()}
    connections = {
        f"{p1}@{mode}": f"{p2}@{mode}"
        for p1, p2 in netlist.connections.items()
        for mode in modes
    }
    ports = {
        f"{p1}@{mode}": f"{p2}@{mode}"
        for p1, p2 in netlist.ports.items()
        for mode in modes
    }
    return connections, ports, models


# -

double_mzi, info = circuit(recnet, models)
double_mzi()


# sometimes it's useful to get the required circuit model names to be able to create the circuit:

# +
# export

def get_required_circuit_models(
        netlist: Union[Netlist, NetlistDict, RecursiveNetlist, RecursiveNetlistDict],
        models: Optional[Dict[str, Model]] = None,
) -> List:
    if models is None:
        models = {}
    assert isinstance(models, dict)
    netlist = _ensure_recursive_netlist_dict(netlist)
    # TODO: do the following two steps *after* recursive netlist parsing.
    netlist = remove_unused_instances(netlist)
    netlist, instance_models = _extract_instance_models(netlist)
    recnet: RecursiveNetlist = _validate_net(netlist)

    missing_models = {}
    missing_model_names = []
    g = nx.DiGraph()

    for model_name, subnetlist in recnet.dict()["__root__"].items():
        if not model_name in missing_models:
            missing_models[model_name] = models.get(model_name, subnetlist)
            g.add_node(model_name)
        if model_name in models:
            continue
        for instance in subnetlist["instances"].values():
            component = instance["component"]
            if not component in missing_models:
                missing_models[component] = models.get(component, None)
                missing_model_names.append(component)
                g.add_node(component)
            g.add_edge(model_name, component)
    return missing_model_names


# -

get_required_circuit_models(recnet, models)
