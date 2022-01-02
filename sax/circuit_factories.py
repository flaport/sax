import os
import re

from functools import partial


from yaml import load as load_yaml, Loader

from sax import models as m
from sax.utils import merge_dicts, validate_model
from sax.core import circuit

from sax._typing import Model
from typing import Optional, Dict, Union, Iterable, Tuple, List

Module = type(m)
Models = Union[Module, Dict[str, Model], List[Union[Module, Dict[str, Model]]]]


def is_model(model: Model):
    if not callable(model):
        return False
    try:
        validate_model(model)
        return True
    except ValueError:
        return False


def circuit_from_netlist(
    netlist: Dict,
    models: Optional[Models] = None,
    auto_prune: bool = False,
    keep: Optional[Iterable[Tuple[str, str]]] = None,
):
    """Load a sax circuit from a GDSFactory netlist

    Args:
        netlist: the GDSFactory netlist to convert to a SAX circuit
        models: a module where all components are defined OR a dictionary
            mapping between component names and SAX models OR a list of multiple
            modules/dict mappings.
        auto_prune: remove zero-valued connections and connections between
            non-output ports *while* evaluating the circuit SDict. This results in
            noticeably better performance and lower memory usage.  However, it also
            makes the resulting circuit non-jittable!
        keep: output port combinations to keep. All other combinations will be
            removed from the final sdict. Note: only output ports specified as
            *values* in the ports dict will be considered. For any port combination
            given, the reciprocal equivalent will automatically be added. This flag
            can be used in stead of ``auto_prune=True`` with jax.jit if you know in
            advance which port combinations of the sdict you're interested in.

    Returns:
        the sax circuit
    """
    models = models if models is not None else {}
    if isinstance(models, (dict, Module)):
        models = [models]
    assert models is not None
    assert not isinstance(models, dict)
    assert not isinstance(models, Module)
    models.insert(0, m)
    for i, _models in enumerate(models):
        if isinstance(_models, Module):
            _models = {k: v for k, v in _models.__dict__.items() if is_model(v)}
        models[i] = _models

    merged_models = merge_dicts(*models)  # type: ignore

    instances = {}
    for instance_name, kwargs in netlist["instances"].items():
        settings = kwargs.get("settings", {})
        params = kwargs.get("params", {})
        settings = merge_dicts(settings, params)
        for k, v in settings.items():
            try:
                settings[k] = float(v)
            except ValueError:
                pass
        model_name = kwargs["component"]
        if model_name in merged_models:
            model = merged_models[model_name]
        elif hasattr(m, model_name):
            model = getattr(m, model_name)
        else:
            raise ValueError(
                f"Could not find a model factory for '{model_name}' (instance '{instance_name}')"
            )
        instance = partial(model, **settings)
        instance_name = re.sub("[^0-9a-zA-Z]", "_", instance_name)
        instances[instance_name] = instance

    connections = {}
    for left, right in netlist["connections"].items():
        left_name, left_port = left.split(",")
        right_name, right_port = right.split(",")
        left = f"{left_name.strip()}:{left_port.strip()}"
        right = f"{right_name.strip()}:{right_port.strip()}"
        left = re.sub("[^0-9a-zA-Z:]", "_", left)
        right = re.sub("[^0-9a-zA-Z:]", "_", right)
        connections[left] = right

    ports = {}
    for left, right in netlist["ports"].items():
        right_name, right_port = right.split(",")
        right = f"{right_name.strip()}:{right_port.strip()}"
        left = re.sub("[^0-9a-zA-Z:]", "_", left)
        right = re.sub("[^0-9a-zA-Z:]", "_", right)
        ports[left] = right

    _circuit = circuit(
        instances=instances,
        connections=connections,
        ports=ports,
        auto_prune=auto_prune,
        keep=keep,
    )

    return _circuit


def circuit_from_yaml(
    yaml: str,
    models: Optional[Models] = None,
    auto_prune: bool = False,
    keep: Optional[Iterable[Tuple[str, str]]] = None,
):
    """Load a sax circuit from yaml definition

    Args:
        yaml: the yaml string to load
        models: a module where all components are defined OR a dictionary
            mapping between component names and SAX models OR a list of multiple
            modules/dict mappings.
        auto_prune: remove zero-valued connections and connections between
            non-output ports *while* evaluating the circuit SDict. This results in
            noticeably better performance and lower memory usage.  However, it also
            makes the resulting circuit non-jittable!
        keep: output port combinations to keep. All other combinations will be
            removed from the final sdict. Note: only output ports specified as
            *values* in the ports dict will be considered. For any port combination
            given, the reciprocal equivalent will automatically be added. This flag
            can be used in stead of ``auto_prune=True`` with jax.jit if you know in
            advance which port combinations of the sdict you're interested in.

    Returns:
        the sax circuit
    """
    if os.path.exists(yaml):
        yaml = os.path.abspath(yaml)
        if os.path.isdir(yaml):
            raise IsADirectoryError(
                f"Cannot read from yaml path. '{yaml}' is a directory"
            )
        yaml = open(yaml, "r").read()

    netlist = load_yaml(yaml, Loader)
    return circuit_from_netlist(
        netlist, models=models, auto_prune=auto_prune, keep=keep
    )


def circuit_from_gdsfactory(
    component,
    models: Optional[Models] = None,
    auto_prune: bool = False,
    keep: Optional[Iterable[Tuple[str, str]]] = None,
):
    """Load a sax circuit from yaml definition

    Args:
        component: the gdsfactory component to convert to a SAX circuit
        models: a module where all components are defined OR a dictionary
            mapping between component names and SAX models OR a list of multiple
            modules/dict mappings.
        auto_prune: remove zero-valued connections and connections between
            non-output ports *while* evaluating the circuit SDict. This results in
            noticeably better performance and lower memory usage.  However, it also
            makes the resulting circuit non-jittable!
        keep: output port combinations to keep. All other combinations will be
            removed from the final sdict. Note: only output ports specified as
            *values* in the ports dict will be considered. For any port combination
            given, the reciprocal equivalent will automatically be added. This flag
            can be used in stead of ``auto_prune=True`` with jax.jit if you know in
            advance which port combinations of the sdict you're interested in.

    Returns:
        the sax circuit
    """
    return circuit_from_netlist(
        component.get_netlist(), models=models, auto_prune=auto_prune, keep=keep
    )
