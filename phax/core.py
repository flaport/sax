import functools
import collections

import jax
import jax.numpy as jnp

float = jnp.float32
complex = jnp.complex64


def model(ports, reciprocal=None, jit=False):
    """decorator for model functions

    Args:
        ports: the port names
        reciprocal: whether the model is reciprocal or not, i.e. whether model(i, j) == model(j, i).
            If a model is reciprocal, the decorated model function only needs to be defined for i <= j.
            Defaults to True for raw model functions. Defaults to the parent value if decorating a model function.
        jit: jit-compile the model. Note: jit-compiling a model is only recommended for the top-level circuit!

    Returns:
        the wrapped model function
    """
    ports = tuple(p for p in ports)
    num_ports = len(ports)

    def model(func):
        assert num_ports == len(set(ports)), f"Duplicate ports found for model {func.__name__}. Got: {ports}"
        assert all(isinstance(p, str) for p in ports), f"Model ports should be string values. Got: {ports}"

        _circuit = False
        _reciprocal = reciprocal
        if hasattr(func, "modelfunc"):
            _circuit = func.circuit
            if _reciprocal is None:
                _reciprocal = func.reciprocal
            func = func.modelfunc
        if _reciprocal is None:
            _reciprocal = True

        @functools.wraps(func)
        def wrapped(params, env, i, j):
            if isinstance(i, str):
                assert i in ports, f"Port name {i} not found in ports {ports} of model {func.__name__}"
                i = ports.index(i)
            if isinstance(j, str):
                assert j in ports, f"Port name {j} not found in ports {ports} of model {func.__name__}"
                j = ports.index(j)
            assert i >= 0, f"First index of {func.__name__} should be bigger or equal to zero."
            assert j >= 0, f"Second index of {func.__name__} should be bigger or equal to zero."
            assert i < num_ports, f"First index of {func.__name__} should be smaller than {num_ports}."
            assert j < num_ports, f"Second index of {func.__name__} should be smaller than {num_ports}."
            if reciprocal and i > j:
                i, j = j, i
            return jnp.asarray(func(params, env, i, j), dtype=complex)

        if jit:
            wrapped = jax.jit(wrapped, static_argnums=(2,3))

        wrapped.modelfunc = func
        wrapped.circuit = _circuit
        wrapped.reciprocal = _reciprocal
        wrapped.num_ports = num_ports
        wrapped.ports = ports
        return wrapped

    return model

_model = model
_component = collections.namedtuple("component", ("params", "env", "model"))
def component(model, params, ports=None, env=None, jit=False):
    """create a component function from a model function

    Args:
        model: model function to create a component for
        params: The parameter dictionary for the model. All necessary model parameters should be present.
        ports: override port names of the model.
        env: the dictionary containing the simulation environment.
        jit: jit-compile the component function. Note: jit-compiling a component model is only recommended for the top-level circuit!

    Returns:
        The partially applied component model with the given model params and port names.

    Note:
        a component is functionally very similar to a functools.partial of model
        with the addition of port names.
    """
    if ports is None:
        ports = model.ports
    else:
        ports = tuple(p for p in ports)

    assert len(ports) == model.num_ports, f"len({ports}) != {model.num_ports}"

    model = _model(ports, reciprocal=model.reciprocal, jit=jit)(model)

    return _component(params, env, model)


def circuit(components, connections, ports, env=None, jit=False):
    """create a (sub)circuit from a collection of components and connections

    Args:
        components: a dictionary with the keys the component names and values
            the component functions
        connections: a dictionary where both keys and values are strings of the
            form "componentname:portname"
        ports: a dictionary mapping portnames of the form
            "componentname:portname" to new unique portnames
        jit: jit-compile the circuit. Note: jit-compiling a circuit model is only recommended for the top-level circuit!

    Returns:
        the circuit component model with the given port names.

    Example:
        A simple add-drop filter can be created as follows::

            waveguide = component(
                model=model_waveguide,
                params={"length": 25e-6, "wl0": 1.55e-6, "neff0": 2.86, "ng": 3.4, "loss": 0.0},
                ports=["in", "out"]
            )
            directional_coupler = component(
                model=model_directional_coupler,
                params={"coupling": 0.3},
                ports=["p0", "p1", "p2", "p3"],
            )
            add_drop = circuit(
                components={
                    "wg1": waveguide,
                    "wg2": waveguide,
                    "dc1": directional_coupler,
                    "dc2": directional_coupler,
                },
                connections={
                    "dc1:p2" : "wg1:in",
                    "wg1:out": "dc2:p1",
                    "dc2:p0" : "wg2:in",
                    "wg2:out": "dc1:p3",
                },
                ports={
                    "dc1:p0": "in",
                    "dc1:p1": "thru",
                    "dc2:p2": "add",
                    "dc2:p3": "drop",
                }
            )
    """
    components, connections, ports = _validate_circuit_parameters(components, connections, ports)

    for name, comp in components.items():
        components[name] = component(comp.model, comp.params, env=None, ports=tuple(f"{name}:{port}" for port in comp.model.ports), jit=False)

    for port1, port2 in connections.items():
        name1, _ = port1.split(":")
        name2, _ = port2.split(":")
        comp1, comp2 = components[name1], components[name2]
        if comp1 != comp2:
            comp = _block_diag_components(comp1, comp2, name1, name2, ports=comp1.model.ports + comp2.model.ports)
        else:
            comp = comp1
        comp = _interconnect_component(comp, port1, port2)
        components = {
            name: (comp if (_comp == comp1 or _comp == comp2) else _comp)
            for name, _comp in components.items()
        }
    return component(comp.model, comp.params, env=env, ports=tuple(ports[port] for port in comp.model.ports), jit=jit)

def _validate_circuit_parameters(components, connections, ports):
    """ validate the netlist parameters of a circuit

    Args:
        components: a dictionary with the keys the component names and values
            the component functions
        connections: a dictionary where both keys and values are strings of the
            form "componentname:portname"
        ports: a dictionary mapping portnames of the form
            "componentname:portname" to new unique portnames
    """
    all_ports = set()
    for name, comp in components.items():
        assert isinstance(comp, tuple), f"Component '{comp}' should be a length-3 tuple: (params, env, model). Please use the component function to construct your component from a model function."
        assert len(comp) == 3, f"Component '{comp}' should be a length-3 tuple: (params, env, model). Please use the component function to construct your component from a model function"
        if not isinstance(comp, _component):
            components[name] = component(comp[2], comp[0], env=None, ports=comp[2].ports, jit=False)
        for port in components[name].model.ports:
            all_ports.add(f"{name}:{port}")

    if not isinstance(connections, dict):
        assert all(len(conn) == 2 for conn in connections), f"Circuit connections should be a str:str dict or a list of length-2 tuples."
        connections, _connections = {k: v for k, v in connections}, connections
        assert len(connections) == len(_connections), f"Duplicate ports found in connection list of tuples"

    connection_ports = set()
    for connection in connections.items():
        for port in connection:
            if port in all_ports:
                all_ports.remove(port)
            assert isinstance(port, str), f"The ports in the connections dictionary should all be strings. Got: '{port}'"
            assert len(port.split(":")) == 2, f"The ports in the connections dictionary should all be of the format 'comp:port'. Got: '{port}'"
            name, _port = port.split(":")
            assert name in components, f"Component '{name}' used in connection '{connection[0]}':'{connection[1]}', but '{name}' not found in components dictionary."
            assert _port in components[name].model.ports, f"Port name '{_port}' not found in component '{name}'. Allowed ports for '{name}': {components[name].model.ports}"
        assert connection[0] not in connection_ports, f"Duplicate port found in connections: '{port}'"
        connection_ports.add(connection[0])
        assert connection[1] not in connection_ports, f"Duplicate port found in connections: '{port}'"
        connection_ports.add(connection[1])

    output_ports = set()
    for port, output_port in ports.items():
        if port in all_ports:
            all_ports.remove(port)
        assert isinstance(port, str), f"The ports in the ports dictionary should all be strings. Got: '{port}'"
        assert isinstance(output_port, str), f"The output ports in the ports dictionary should all be strings. Got: '{output_port}'"
        assert len(port.split(":")) == 2, f"The ports in the ports dictionary should all be of the format 'comp:port'. Got: '{port}'"
        assert ':' not in output_port, f"The output ports in the ports dictionary cannot contain a colon (':'). Got: '{output_port}'"
        assert port not in connection_ports, f"Duplicate port found in ports or connections: '{port}'"
        name, _port = port.split(":")
        assert name in components, f"Component '{name}' used in output_port '{port}':'{output_port}', but '{name}' not found in components dictionary."
        assert _port in components[name].model.ports, f"Port name '{_port}' not found in component '{name}'. Allowed ports for '{name}': {components[name].model.ports}"
        connection_ports.add(port)
        assert output_port not in output_ports, f"Duplicate port found in output ports: '{output_port}'"
        output_ports.add(output_port)

    assert not all_ports, f"Unused ports found: {all_ports}"

    return components, connections, ports

def _block_diag_components(comp1, comp2, name1, name2, ports=None):
    """combine two components as if their S-matrices were stacked block-diagonally

    Args:
        comp1: the first component to combine
        comp2: the second component to combine
        name1: the name of the first component
        name2: the name of the second component
        ports: new port names for the combined component.
            If no port names are given, the port names of the components will attempted to be used.
            If this results in duplicate port names, all port names will silently be relabeled as 'p{i}'.

    Returns:
        the merged 'block-diagonal' component.
    """
    num_ports = comp1.model.num_ports + comp2.model.num_ports

    if ports is None:
        ports = comp1.model.ports + comp2.model.ports
    else:
        ports = tuple(ports)
    if len(ports) < len(set(ports)):
        ports = tuple(f"p{i}" for i in range(comp1.num_ports + comp2.num_ports))

    @model(ports=ports, reciprocal=False, jit=False)
    def model_block_diag(params, env, i, j):
        if i < comp1.model.num_ports and j < comp1.model.num_ports:
            if comp1.model.circuit:
                return comp1.model(params, env, i, j)
            else:
                return comp1.model(params[name1], env, i, j)
        elif i >= comp1.model.num_ports and j >= comp1.model.num_ports:
            if comp2.model.circuit:
                return comp2.model(params, env,  i - comp1.model.num_ports, j - comp1.model.num_ports)
            else:
                return comp2.model(params[name2], env,  i - comp1.model.num_ports, j - comp1.model.num_ports)
        else:
            return 0
    model_block_diag.circuit = True

    if comp1.model.circuit and comp2.model.circuit:
        params = {**comp1.params, **comp2.params}
    elif comp1.model.circuit:
        params = {**comp1.params}
        params[name2] = comp2.params
    elif comp2.model.circuit:
        params = {**comp2.params}
        params[name1] = comp1.params
    else:
        params = {
            name1: comp1.params,
            name2: comp2.params,
        }

    return component(model_block_diag, params, ports=ports)

def _interconnect_component(comp, k, l, ports=None):
    """interconnect two ports in a given component

    Args:
        comp: the first component to interconnect
        k: the first index to connect
        l: the second index to connect
        ports: new port names for the combined component.
            If no port names are given, the original remaining port names will be used.
            If any duplicate port names are given, all port names will silently be relabled as 'p{i}'

    Returns:
        the resulting interconnected component, i.e. a component with two ports less than the original component.

    Note:
        The interconnect algorithm below is based on equation 6 in the paper below::

            Filipsson, Gunnar. "A new general computer algorithm for S-matrix calculation
            of interconnected multiports." 11th European Microwave Conference. IEEE, 1981.
    """
    num_ports = comp.model.num_ports - 2
    if isinstance(k, str):
        k = comp.model.ports.index(k)
    if isinstance(l, str):
        l = comp.model.ports.index(l)

    if ports is None:
        ports = tuple(port for i, port in enumerate(comp.model.ports) if i not in (k, l))
    else:
        ports = tuple(ports)
    if len(ports) < len(set(ports)):
        ports = tuple(f"p{i}" for i in range(num_ports))

    @model(ports=ports, reciprocal=False, jit=False)
    def model_interconnected(params, env, i, j):
        if k < l:
            if i >= k:
                i += 1
            if i >= l:
                i += 1
            if j >= k:
                j += 1
            if j >= l:
                j += 1
        if l < k:
            if i >= l:
                i += 1
            if i >= k:
                i += 1
            if j >= l:
                j += 1
            if j >= k:
                j += 1
        m = functools.lru_cache(maxsize=4096)(functools.partial(comp.model, params, env))
        return m(i, j) + (
            m(k, j) * m(i, l) * (1 - m(l, k))
            + m(l, j) * m(i, k) * (1 - m(k, l))
            + m(k, j) * m(l, l) * m(i, k)
            + m(l, j) * m(k, k) * m(i, l)
        ) / ((1 - m(k, l)) * (1 - m(l, k)) - m(k, k) * m(l, l))
    model_interconnected.circuit = comp.model.circuit

    return component(model_interconnected, comp.params, ports=ports)
