import functools

import jax
import jax.numpy as jnp

float = jnp.float32
complex = jnp.complex64


def model(num_ports, reciprocal=True):
    """decorator for model functions

    Args:
        num_ports: number of ports of the model
        reciprocal: whether the model is reciprocal or not, i.e. whethere model(i, j) == model(j, i).
            If a model is reciprocal, the decorated model function only needs to be defined for i <= j.

    Returns:
        the wrapped model function
    """

    def model(func):
        @functools.wraps(func)
        def wrapped(params, env, i, j):
            if reciprocal and i > j:
                i, j = j, i
            return jnp.asarray(func(params, env, i, j), dtype=complex)

        wrapped.func = func
        wrapped.reciprocal = reciprocal
        wrapped.num_ports = num_ports
        return wrapped

    return model


def component(model, params, ports=None, jit=True):
    """create a component function from a model function

    Args:
        model: model function to create a component for
        params: The parameter dictionary for the model. All necessary model parameters should be present.
        ports: port names for each of the model indices
        jit: jit-compile the component function

    Returns:
        The partially applied component model with the given model params and port names.

    Note:
        a component is functionally very similar to a functools.partial of model
        with the addition of port names.
    """
    partial_model = functools.partial(model, params)
    if hasattr(model, "ports") and ports is None:
        ports = tuple(model.ports)
    elif ports is None:
        ports = tuple(f"p{i}" for i in range(model.num_ports))
    else:
        if len(ports) != model.num_ports:
            raise ValueError(
                f"Number of ports given ({len(ports)}) is different from the expected "
                f"number of ports in the model ({model.num_ports})."
            )
        ports = tuple(ports)

    @functools.wraps(partial_model)
    def component(env, i, j):
        if isinstance(i, str):
            i = ports.index(i)
        if isinstance(j, str):
            j = ports.index(j)
        return partial_model(env, i, j)

    if jit:
        component = jax.jit(component, static_argnums=(1,2))
    component.__name__ = model.__name__.replace("model_", "")
    component.__qualname__ = model.__qualname__.replace("model_", "")
    component.model = model
    component.params = params
    component.ports = ports
    component.num_ports = len(ports)
    return component


def circuit(components, connections, ports, jit=True):
    """create a (sub)circuit from a collection of components and connections

    Args:
        components: a dictionary with the keys the component names and values
            the component functions
        connections: a dictionary where both keys and values are strings of the
            form "componentname:portname"
        ports: a dictionary mapping portnames of the form
            "componentname:portname" to new unique portnames
        jit: jit-compile the circuit

    Returns:
        the circuit component model with the given port names.

    Example:
        A simple add-drop filter can be created as follows::

            waveguide = component(model_waveguide, ["in", "out"], length=25e-6)
            directional_coupler = component(
                model_directional_coupler, ["p0", "p1", "p2", "p3"], coupling=0.3
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
    for name, comp in components.items():
        components[name] = component(
            comp.model, comp.params, ports=tuple(f"{name}:{port}" for port in comp.ports), jit=False
        )
    for port1, port2 in connections.items():
        name1, _ = port1.split(":")
        name2, _ = port2.split(":")
        comp1, comp2 = components[name1], components[name2]
        if comp1 != comp2:
            comp = _block_diag_components(comp1, comp2, ports=comp1.ports + comp2.ports)
        else:
            comp = comp1
        comp = _interconnect_component(comp, port1, port2)
        components = {
            name: (comp if (_comp == comp1 or _comp == comp2) else _comp)
            for name, _comp in components.items()
        }
    return component(comp.model, comp.params, ports=tuple(ports[port] for port in comp.ports), jit=jit)


def _block_diag_components(comp1, comp2, ports=None):
    """combine two components as if their S-matrices were stacked block-diagonally

    Args:
        comp1: the first component to combine
        comp2: the second component to combine
        ports: new port names for the combined component.
            If no port names are given, the port names of the components will attempted to be used.
            If this results in duplicate port names, all port names will silently be relabeled as 'p{i}'.

    Returns:
        the merged 'block-diagonal' component.
    """
    num_ports = comp1.num_ports + comp2.num_ports

    @model(num_ports=num_ports)
    def model_stacked(params, env, i, j):
        if i < comp1.num_ports and j < comp1.num_ports:
            return comp1.model(params[0], env, i, j)
        elif i >= comp1.num_ports and j >= comp1.num_ports:
            return comp2.model(params[1], env,  i - comp1.num_ports, j - comp1.num_ports)
        else:
            return 0

    if ports is None:
        ports = comp1.ports + comp2.ports
    else:
        ports = tuple(ports)
    if len(ports) < len(set(ports)):
        ports = tuple(f"p{i}" for i in range(comp1.num_ports + comp2.num_ports))
        
    return component(model_stacked, (comp1.params, comp2.params), ports=ports)


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
    num_ports = comp.num_ports - 2
    if isinstance(k, str):
        k = comp.ports.index(k)
    if isinstance(l, str):
        l = comp.ports.index(l)

    @model(num_ports=num_ports)
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

    if ports is None:
        ports = tuple(port for i, port in enumerate(comp.ports) if i not in (k, l))
    else:
        ports = tuple(ports)
    if len(ports) < len(set(ports)):
        ports = tuple(f"p{i}" for i in range(num_ports))
    return component(model_interconnected, comp.params, ports=ports)
