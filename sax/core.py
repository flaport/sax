""" SAX core """

from __future__ import annotations

import functools

from .utils import zero, rename_ports, get_ports, validate_params, copy_params
from .typing import (
    Optional,
    Callable,
    Tuple,
    Dict,
    ModelParams,
    ModelDict,
    Model,
    ModelFunc,
    ComplexFloat,
)


def model(funcs: ModelDict, params: ModelParams) -> Model:
    """create a model dictionary

    Args:
        funcs: a dictionary mapping port name combinations onto functions
        params: a parameter dictionary
    """
    return Model(funcs=funcs, params=params)


def circuit(
    models: Dict[str, Model], connections: Dict[str, str], ports: Dict[str, str]
) -> Model:
    """create a (sub)circuit model from a collection of models and connections

    Args:
        models: a dictionary with as keys the model names and values
            the model dictionaries.
        connections: a dictionary where both keys and values are strings of the
            form "modelname:portname"
        ports: a dictionary mapping portnames of the form
            "modelname:portname" to new unique portnames

    Returns:
        the circuit model dictionary with the given port names.

    Example:
        A simple mzi can be created as follows::

            import sax
            mzi = sax.circuit(
                models = {
                    "left": sax.models.pic.dc,
                    "top": sax.models.pic.wg,
                    "bottom": sax.models.pic.wg,
                    "right": sax.models.pic.dc,
                },
                connections={
                    "left:p2": "top:in",
                    "left:p1": "bottom:in",
                    "top:out": "right:p3",
                    "bottom:out": "right:p0",
                },
                ports={
                    "left:p3": "in2",
                    "left:p0": "in1",
                    "right:p2": "out2",
                    "right:p1": "out1",
                },
            )
    """

    models, connections, ports = _validate_circuit_parameters(
        models, connections, ports
    )

    for name, model in models.items():
        models[name] = rename_ports(model, {p: f"{name}:{p}" for p in get_ports(model)})
        validate_params(models[name].params)
    modelnames = [[name] for name in models]

    model: Model = Model(funcs={}, params={})
    while len(modelnames) > 1:
        for names1, names2 in zip(modelnames[::2], modelnames[1::2]):
            model1 = models.pop(names1[0])
            model2 = models.pop(names2[0])
            model = _combine_models(
                model1,
                model2,
                None if len(names1) > 1 else names1[0],
                None if len(names2) > 1 else names2[0],
            )
            names1.extend(names2)
            for port1, port2 in [(k, v) for k, v in connections.items()]:
                n1, _ = port1.split(":")
                n2, _ = port2.split(":")
                if n1 in names1 and n2 in names1:
                    del connections[port1]
                    model = _interconnect_model(model, port1, port2)
            models[names1[0]] = model
        modelnames = list(reversed(modelnames[::2]))

    if model.funcs:
        model = rename_ports(model, ports)

    return model


def _validate_circuit_parameters(
    models: Dict[str, Model], connections: Dict[str, str], ports: Dict[str, str]
) -> Tuple[Dict[str, Model], Dict[str, str], Dict[str, str]]:
    """validate the netlist parameters of a circuit

    Args:
        models: a dictionary with as keys the model names and values
            the model dictionaries.
        connections: a dictionary where both keys and values are strings of the
            form "modelname:portname"
        ports: a dictionary mapping portnames of the form
            "modelname:portname" to new unique portnames

    Returns:
        the validated and possibly slightly modified models, connections and
        ports dictionaries.
    """

    all_ports = set()
    for name, model in models.items():
        _validate_model_dict(name, model)
        for port in get_ports(model):
            all_ports.add(f"{name}:{port}")

    if not isinstance(connections, dict):
        msg = f"Connections should be a str:str dict or a list of length-2 tuples."
        connections, _connections = {}, connections
        connection_ports = set()
        for conn in _connections:
            connections[conn[0]] = conn[1]
            for port in conn:
                msg = f"Duplicate port found in connections: '{port}'"
                assert port not in connection_ports, msg
                connection_ports.add(port)

    connection_ports = set()
    for connection in connections.items():
        for port in connection:
            if port in all_ports:
                all_ports.remove(port)
            msg = f"Connection ports should all be strings. Got: '{port}'"
            assert isinstance(port, str), msg
            msg = f"Connection ports should have format 'modelname:port'. Got: '{port}'"
            assert len(port.split(":")) == 2, msg
            name, _port = port.split(":")
            msg = f"Model '{name}' used in connection "
            msg += f"'{connection[0]}':'{connection[1]}', "
            msg += f"but '{name}' not found in models dictionary."
            assert name in models, msg
            msg = f"Port name '{_port}' not found in model '{name}'. "
            msg += f"Allowed ports for '{name}': {get_ports(models[name])}"
            assert _port in get_ports(models[name]), msg
            msg = f"Duplicate port found in connections: '{port}'"
            assert port not in connection_ports, msg
            connection_ports.add(port)

    output_ports = set()
    for port, output_port in ports.items():
        if port in all_ports:
            all_ports.remove(port)
        msg = f"Ports keys in 'ports' should all be strings. Got: '{port}'"
        assert isinstance(port, str), msg
        msg = f"Port values in 'ports' should all be strings. Got: '{output_port}'"
        assert isinstance(output_port, str), msg
        msg = f"Port keys in 'ports' should have format 'model:port'. Got: '{port}'"
        assert len(port.split(":")) == 2, msg
        msg = f"Port values in 'ports' shouldn't contain a ':'. Got: '{output_port}'"
        assert ":" not in output_port, msg
        msg = f"Duplicate port found in ports or connections: '{port}'"
        assert port not in connection_ports, msg
        name, _port = port.split(":")
        msg = f"Model '{name}' used in output port "
        msg += f"'{port}':'{output_port}', "
        msg += f"but '{name}' not found in models dictionary."
        assert name in models, msg
        msg = f"Port name '{_port}' not found in model '{name}'. "
        msg += f"Allowed ports for '{name}': {get_ports(models[name])}"
        assert _port in get_ports(models[name]), msg
        connection_ports.add(port)
        msg = f"Duplicate port found in output ports: '{output_port}'"
        assert output_port not in output_ports, msg
        output_ports.add(output_port)

    assert not all_ports, f"Unused ports found: {all_ports}"

    return models, connections, ports


def _validate_model_dict(name: str, model: Model):
    assert isinstance(model, Model), f"'{model}' should be a Model tuple"
    ports = get_ports(model)
    assert ports, f"No ports in model {name}"
    for p1 in ports:
        for p2 in ports:
            msg = (
                f"model {name} port combination {p1}->{p2} is no function or callable."
            )
            assert callable(model.funcs.get((p1, p2), zero)), msg


def _namedparamsfunc(func: Callable, name: str, params: ModelParams) -> ComplexFloat:
    """make a model function look for its model name before acting on the parameters

    Args:
        func: the original model function acting on a dictionary of parameters
        name: the name of the model
        params: a dictionary for which the keys are the model names and the
            values are the model parameter dictionaries.
    """
    return func(params[name])


def _combine_models(
    model1: Model,
    model2: Model,
    name1: Optional[str] = None,
    name2: Optional[str] = None,
) -> Model:
    """Combine two models into a combined model (without connecting any ports)

    Args:
        model1: the first model dictionary to combine
        model2: the second model dictionary to combine
        name1: the name of the first model (can be None for unnamed models)
        name2: the name of the second model (can be None for unnamed models)
    """
    funcs: ModelDict = {}
    params: ModelParams = {}
    for _model, _name in [(model1, name1), (model2, name2)]:
        for key, value in _model.funcs.items():
            p1, p2 = key
            _params = copy_params(_model.params)
            if value is zero or _name is None:
                funcs[p1, p2] = value
            else:
                funcs[p1, p2] = _partialmodelfunc(_namedparamsfunc, value, _name)
            if _name is None:
                params = {**params, **_params}
            else:
                params[_name] = _params
    return Model(funcs=funcs, params=params)


def _interconnect_model(model: Model, k: str, l: str) -> Model:
    """interconnect two ports in a given model

    Args:
        model: the component for which to interconnect the given ports
        k: the first port name to connect
        l: the second port name to connect

    Returns:
        the resulting interconnected component, i.e. a component with two ports
        less than the original component.

    Note:
        The interconnect algorithm is based on equation 6 in the paper below::

          Filipsson, Gunnar. "A new general computer algorithm for S-matrix calculation
          of interconnected multiports." 11th European Microwave Conference. IEEE, 1981.
    """
    funcs: ModelDict = {}
    ports = get_ports(model)
    for i in ports:
        for j in ports:
            mij = model.funcs.get((i, j), zero)
            mik = model.funcs.get((i, k), zero)
            mil = model.funcs.get((i, l), zero)
            mkj = model.funcs.get((k, j), zero)
            mkk = model.funcs.get((k, k), zero)
            mkl = model.funcs.get((k, l), zero)
            mlj = model.funcs.get((l, j), zero)
            mlk = model.funcs.get((l, k), zero)
            mll = model.funcs.get((l, l), zero)
            if (
                (mij is zero)
                and ((mkj is zero) or (mil is zero))
                and ((mlj is zero) or (mik is zero))
                and ((mkj is zero) or (mll is zero) or (mik is zero))
                and ((mlj is zero) or (mkk is zero) or (mil is zero))
            ):
                continue
            funcs[i, j] = _partialmodelfunc(
                _model_ijkl, mij, mik, mil, mkj, mkk, mkl, mlj, mlk, mll
            )
    for key in list(funcs.keys()):
        i, j = key
        if i == k or i == l or j == k or j == l:
            del funcs[i, j]
    return Model(funcs=funcs, params=copy_params(model.params))


def _model_ijkl(
    mij: ModelFunc,
    mik: ModelFunc,
    mil: ModelFunc,
    mkj: ModelFunc,
    mkk: ModelFunc,
    mkl: ModelFunc,
    mlj: ModelFunc,
    mlk: ModelFunc,
    mll: ModelFunc,
    params: ModelParams,
) -> ComplexFloat:
    """combine the given model functions.

    Note:
        The interconnect algorithm is based on equation 6 in the paper below::

          Filipsson, Gunnar. "A new general computer algorithm for S-matrix calculation
          of interconnected multiports." 11th European Microwave Conference. IEEE, 1981.
    """
    vij = mij(params)
    vik = mik(params)
    vil = mil(params)
    vkj = mkj(params)
    vkk = mkk(params)
    vkl = mkl(params)
    vlj = mlj(params)
    vlk = mlk(params)
    vll = mll(params)
    return vij + (
        vkj * vil * (1 - vlk)
        + vlj * vik * (1 - vkl)
        + vkj * vll * vik
        + vlj * vkk * vil
    ) / ((1 - vkl) * (1 - vlk) - vkk * vll)


class _partialmodelfunc(functools.partial):
    """fun(params)

    Args:
        params: parameter dictionary for the model.

    Returns:
        Model transmission.
    """

    def __repr__(self):
        func = self.func
        while hasattr(func, "func"):
            func = func.func
        return repr(func)
