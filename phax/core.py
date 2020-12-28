import functools
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
        def wrapped(i, j, **kwargs):
            if reciprocal and i > j:
                i, j = j, i
            return jnp.asarray(func(i, j, **kwargs), dtype=complex)

        wrapped.num_ports = num_ports
        return wrapped

    return model


def component(model, ports=None, **kwargs):
    """create a component function from a model function

    Args:
        model: model function to create a component for
        ports: port names for each of the model indices
        **kwargs: the keyword arguments to partially apply into the supplied model.

    Returns:
        The partially applied component model with the given port names.

    Note:
        a component is functionally very similar to a functools.partial of model
        with the addition of port names.
    """
    partial_model = functools.partial(model, **kwargs)
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
    def component(i, j, **kwargs):
        if isinstance(i, str):
            i = ports.index(i)
        if isinstance(j, str):
            j = ports.index(j)
        return partial_model(i, j, **kwargs)

    component.__name__ = model.__name__.replace("model_", "")
    component.__qualname__ = model.__qualname__.replace("model_", "")
    component.ports = ports
    component.num_ports = len(ports)
    return component


def circuit(components, connections, ports):
    """create a (sub)circuit from a collection of components and connections

    Args:
        components: a dictionary with the keys the component names and values
            the component functions
        connections: a dictionary where both keys and values are strings of the
            form "componentname:portname"
        ports: a dictionary mapping portnames of the form
            "componentname:portname" to new unique portnames

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
            comp, ports=tuple(f"{name}:{port}" for port in comp.ports)
        )
    for port1, port2 in connections.items():
        name1, _ = port1.split(":")
        name2, _ = port2.split(":")
        comp1, comp2 = components[name1], components[name2]
        if comp1 != comp2:
            comp = _block_diag_models(comp1, comp2, ports=comp1.ports + comp2.ports)
        else:
            comp = comp1
        comp = _interconnect_model(comp, port1, port2)
        components = {
            name: (comp if (_comp == comp1 or _comp == comp2) else _comp)
            for name, _comp in components.items()
        }
    return component(comp, ports=tuple(ports[port] for port in comp.ports))


def _block_diag_models(partial_model1, partial_model2, ports=None):
    """combine two models as if their S-matrices were stacked block-diagonally

    Args:
        partial_model1: the first model to combine
        partial_model2: the second model to combine
        ports: new port names for the combined model.
            If no port names are given, the port names of the partial_models will attempted to be used.
            If this results in duplicate port names, the port names will silently be relabeled as 'p{i}'.

    Returns:
        the merged 'block-diagonal' model.
    """
    num_ports = partial_model1.num_ports + partial_model2.num_ports

    @model(num_ports=num_ports)
    def model_stacked(i, j, wl=1.55e-6):
        if i < partial_model1.num_ports and j < partial_model1.num_ports:
            return partial_model1(i, j, wl=wl)
        elif i >= partial_model1.num_ports and j >= partial_model1.num_ports:
            return partial_model2(
                i - partial_model1.num_ports, j - partial_model1.num_ports, wl=wl
            )
        else:
            return 0

    if ports is None:
        ports = tuple(
            f"p{i}" for i in range(partial_model1.num_ports + partial_model2.num_ports)
        )
    else:
        ports = tuple(ports)
    if len(ports) < len(set(ports)):
        ports = tuple(
            f"p{i}" for i in range(partial_model1.num_ports + partial_model2.num_ports)
        )
    model_stacked.ports = ports
    return model_stacked


def _interconnect_model(partial_model, k, l, ports=None):
    """interconnect two ports in a given model

    Args:
        partial_model: the first model to interconnect
        k: the first index to connect
        l: the second index to connect
        ports: new port names for the combined model.
            If no port names are given, the original remaining port names will be used.
            If any duplicate port names are given, the port names will silently be relabled as 'p{i}'

    Returns:
        the resulting interconnected model, i.e. a model with two ports less than the original model.

    Note:
        The interconnect algorithm below is based on equation 6 in the paper below::

            Filipsson, Gunnar. "A new general computer algorithm for S-matrix calculation
            of interconnected multiports." 11th European Microwave Conference. IEEE, 1981.
    """
    num_ports = partial_model.num_ports - 2
    if isinstance(k, str):
        k = partial_model.ports.index(k)
    if isinstance(l, str):
        l = partial_model.ports.index(l)

    @model(num_ports=num_ports)
    def model_interconnected(i, j, wl=1.55e-6):
        if isinstance(i, str):
            i = partial_model.ports.index(i)
        if isinstance(j, str):
            j = partial_model.ports.index(j)
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
        m = functools.lru_cache(maxsize=4096)(functools.partial(partial_model, wl=wl))
        return m(i, j) + (
            m(k, j) * m(i, l) * (1 - m(l, k))
            + m(l, j) * m(i, k) * (1 - m(k, l))
            + m(k, j) * m(l, l) * m(i, k)
            + m(l, j) * m(k, k) * m(i, l)
        ) / ((1 - m(k, l)) * (1 - m(l, k)) - m(k, k) * m(l, l))

    if ports is None:
        ports = tuple(
            port for i, port in enumerate(partial_model.ports) if i not in (k, l)
        )
    else:
        ports = tuple(ports)
    if len(ports) < len(set(ports)):
        ports = tuple(f"p{i}" for i in range(num_ports))
    model_interconnected.ports = ports
    return model_interconnected
