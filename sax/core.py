""" SAX core """

from __future__ import annotations

from functools import wraps
from itertools import combinations_with_replacement, product

import jax
import jax.numpy as jnp

from sax.utils import get_params, get_ports, validate_model, _replace_kwargs
from typing import (
    Dict,
    Iterable,
    Optional,
    Tuple,
    Union,
    overload,
)
from sax._typing import (
    Model,
    SDict,
)


def circuit(
    instances: Dict[str, Model],
    connections: Dict[str, str],
    ports: Dict[str, str],
    auto_prune: bool = False,
    keep: Optional[Iterable[Tuple[str, str]]] = None,
    modes: Optional[Tuple[str, ...]] = None,
) -> Model:
    """generate a circuit model for the instance models given

    Args:
        instances: a dictionary with as keys the model names and values
            the model dictionaries.
        connections: a dictionary where both keys and values are strings of the
            form "instancename:portname"
        ports: a dictionary mapping output portnames of the circuit to
            portnames of the form "instancename:portname".
        auto_prune: remove zero-valued connections and connections between
            non-output ports *while* evaluating the circuit SDict. This results in
            noticeably better performance and lower memory usage.  However, it also
            makes the resulting circuit non-jittable!
        keep: output port combinations to keep. All other combinations will be
            removed from the final sdict. Note: only output ports specified as
            *keys* in the ports dict will be considered. For any port combination
            given, the reciprocal equivalent will automatically be added. This flag
            can be used in stead of ``auto_prune=True`` with jax.jit if you know in
            advance which port combinations of the sdict you're interested in.
        modes: if given, a multimode simulation will be performed for all the
            modes specified. Any given single-mode instance models will be
            converted into multimode models by assuming no cross_polarization
            and the same S-parameters everywhere.

    Returns:
        the circuit model with the given port names.

    Example:
        A simple mzi can be created as follows::

            import sax
            mzi = sax.circuit(
                instances = {
                    "lft": coupler_model,
                    "top": waveguide_model,
                    "btm": waveguide_model,
                    "rgt": coupler_model,
                },
                connections={
                    "lft:out0": "btm:in0",
                    "btm:out0": "rgt:in0",
                    "lft:out1": "top:in0",
                    "top:out0": "rgt:in1",
                },
                ports={
                    "lft:in0": "in0",
                    "lft:in1": "in1",
                    "rgt:out0": "out0",
                    "rgt:out1": "out1",
                },
            )
    """
    connections = {  # silently accept YAML netlist syntax
        k.replace(",", ":"): v.replace(",", ":") for k, v in connections.items()
    }
    ports = {  # silently accept YAML netlist syntax
        k.replace(",", ":"): v.replace(",", ":") for k, v in ports.items()
    }

    instances, connections, ports = validate_circuit_args(instances, connections, ports)

    if keep:
        keep_dict = {min(p1, p2): max(p1, p2) for p1, p2 in keep}
        keep = tuple((p1, p2) for p1, p2 in keep_dict.items())

    if modes is not None:
        old_instances = instances
        old_connections = connections
        old_ports = ports
        instances, connections, ports = {}, {}, {}
        instances = old_instances
        for name, instance in old_instances.items():
            instances[name] = multimode(instance, modes=modes)
        for p1, p2 in old_connections.items():
            for mode in modes:
                connections[f"{p1}@{mode}"] = f"{p2}@{mode}"
        for p1, p2 in old_ports.items():
            for mode in modes:
                ports[f"{p1}@{mode}"] = f"{p2}@{mode}"
        if keep:
            keep_dict = {}
            for p1, p2 in keep:
                for m1 in modes:
                    for m2 in modes:
                        keep_dict[f"{p1}@{m1}"] = f"{p2}@{m2}"
                        keep_dict[f"{p1}@{m2}"] = f"{p2}@{m1}"
            keep = tuple((p1, p2) for p1, p2 in keep_dict.items())
    else:
        instances = {name: singlemode(model) for name, model in instances.items()}

    def circuit(**params):
        sdicts = {
            name: model(**params.get(name, {})) for name, model in instances.items()
        }
        return _evaluate_circuit(
            sdicts, connections, ports, auto_prune=auto_prune, keep=keep
        )

    params = {name: get_params(model) for name, model in instances.items()}
    modified_circuit = _replace_kwargs(circuit, **params)

    return modified_circuit


@overload
def multimode(
    sdict_or_model: SDict,
    modes: Optional[Tuple[str, ...]] = None,
    backreflection: bool = False,
    cross_polarization: bool = False,
) -> SDict:
    ...


@overload
def multimode(
    sdict_or_model: Model,
    modes: Optional[Tuple[str, ...]] = None,
    backreflection: bool = False,
    cross_polarization: bool = False,
) -> Model:
    ...


def multimode(
    sdict_or_model: Union[SDict, Model],
    modes: Optional[Tuple[str, ...]] = None,
    backreflection: bool = False,
    cross_polarization: bool = False,
) -> Union[SDict, Model]:
    """Convert to a multimode model/sdict

    Args:
        sdict_or_model: the single mode sdict or model to convert to multimode
        modes: the modes of the multimode model (e.g. ["te", "tm"])
        backreflection: whether to add backreflection terms to the multimode model
        cross_polarization: whether to add cross polarization terms to the multimode model.

    Returns:
        the modified sdict or model
    """
    if modes is None:
        return sdict_or_model
    elif isinstance(sdict_or_model, dict):
        sdict = sdict_or_model
        ports = get_ports(sdict)
        new_sdict = {}
        for (p1, m1), (p2, m2) in combinations_with_replacement(
            product(ports, modes), 2
        ):
            if not cross_polarization and m1 != m2:
                continue
            if not backreflection and p1 == p2:
                continue
            pm1 = f"{p1}@{m1}" if "@" not in p1 else p1
            pm2 = f"{p2}@{m2}" if "@" not in p2 else p2
            new_sdict[pm1, pm2] = sdict.get((p1, p2), 0.0)
            new_sdict[pm2, pm1] = sdict.get((p2, p1), 0.0)
        return new_sdict
    else:
        old_model = sdict_or_model

        @wraps(old_model)
        def new_model(**params):
            return multimode(old_model(**params), modes=modes)

        return new_model


@overload
def singlemode(sdict_or_model: SDict, mode: Optional[str] = None) -> SDict:
    ...


@overload
def singlemode(sdict_or_model: Model, mode: Optional[str] = None) -> Model:
    ...


def singlemode(
    sdict_or_model: Union[SDict, Model], mode: Optional[str] = None
) -> Union[SDict, Model]:
    """Convert to a single mode model

    Args:
        sdict_or_model: the multimode sdict or model to convert to single mode
        mode: the mode to select from the multimode model

    Returns:
        the single mode sdict or model

    Note:
        if no mode is selected, the mean of all modes is taken.

    """
    if isinstance(sdict_or_model, dict):
        sdict = sdict_or_model
        new_sdict = {}
        norm = 1.0
        modes = {("" if "@" not in p else p.split("@")[1]) for p, _ in sdict}
        modes |= {("" if "@" not in p else p.split("@")[1]) for _, p in sdict}
        if not (modes - {""}):
            return sdict  # sdict is already single mode
        if mode is None:
            num_modes = len(modes)
            norm = 1.0 / num_modes
        for (pm1, pm2), value in sdict.items():
            p1, *m1 = pm1.split("@")
            p2, *m2 = pm2.split("@")
            if mode is None or (mode in m1 and mode in m2):
                new_sdict[p1, p2] = new_sdict.get((p1, p2), 0.0) + value * norm
        return new_sdict
    else:
        old_model = sdict_or_model

        @wraps(old_model)
        def new_model(**params):
            return singlemode(old_model(**params), mode=mode)

        return new_model


def _evaluate_circuit(
    instances: Dict[str, SDict],
    connections: Dict[str, str],
    ports: Dict[str, str],
    auto_prune: bool = False,
    keep: Optional[Iterable[Tuple[str, str]]] = None,
):
    """evaluate a circuit for the sdicts (instances) given.

    Args:
        instances: a dictionary with as keys the instance names and values
            the corresponding SDicts.
        connections: a dictionary where both keys and values are strings of the
            form "instancename:portname"
        ports: a dictionary mapping output portnames of the circuit to
            portnames of the form "instancename:portname".
        auto_prune: remove zero-valued connections and connections between
            non-output ports *while* evaluating the circuit SDict. This results in
            noticeably better performance and lower memory usage.  However, it also
            makes the resulting circuit non-jittable!
        keep: output port combinations to keep. All other combinations will be
            removed from the final sdict. Note: only output ports specified as
            *keys* in the ports dict will be considered. For any port combination
            given, the reciprocal equivalent will automatically be added. This flag
            can be used in stead of ``auto_prune=True`` with jax.jit if you know in
            advance which port combinations of the sdict you're interested in.

    Returns:
        the circuit model dictionary with the given port names.
    """
    connections = {  # silently accept YAML netlist syntax
        k.replace(",", ":"): v.replace(",", ":") for k, v in connections.items()
    }
    ports = {  # silently accept YAML netlist syntax
        k.replace(",", ":"): v.replace(",", ":") for k, v in ports.items()
    }
    ports = {v: k for k, v in ports.items()}  # it's actually easier working w reverse
    float_eps = 2 * jnp.finfo(jnp.zeros(0, dtype=float).dtype).resolution

    if keep:
        keep = set(list(keep) + [(p2, p1) for p1, p2 in keep])

    block_diag = {}
    for name, sdict in instances.items():
        block_diag.update(
            {(f"{name}:{p1}", f"{name}:{p2}"): v for (p1, p2), v in sdict.items()}
        )

    sorted_connections = sorted(connections.items(), key=_connections_sort_key)
    all_connected_instances = {k: {k} for k in instances}
    for k, l in sorted_connections:
        name1, _ = k.split(":")
        name2, _ = l.split(":")

        connected_instances = (
            all_connected_instances[name1] | all_connected_instances[name2]
        )
        for name in connected_instances:
            all_connected_instances[name] = connected_instances

        current_ports = tuple(
            p
            for instance in connected_instances
            for p in set([p for p, _ in block_diag] + [p for _, p in block_diag])
            if p.startswith(f"{instance}:")
        )

        block_diag.update(_interconnect_ports(block_diag, current_ports, k, l))

        for i, j in list(block_diag.keys()):
            is_connected = i == k or i == l or j == k or j == l
            is_in_output_ports = i in ports and j in ports
            if is_connected and not is_in_output_ports:
                del block_diag[i, j]  # we're not interested in these port combinations

    if auto_prune:
        circuit_sdict = {
            (ports[i], ports[j]): v
            for (i, j), v in block_diag.items()
            if i in ports and j in ports and jnp.any(jnp.abs(v) > float_eps)
        }
    elif keep:
        circuit_sdict = {
            (ports[i], ports[j]): v
            for (i, j), v in block_diag.items()
            if i in ports and j in ports and (ports[i], ports[j]) in keep
        }
    else:
        circuit_sdict = {
            (ports[i], ports[j]): v
            for (i, j), v in block_diag.items()
            if i in ports and j in ports
        }
    return circuit_sdict


def _connections_sort_key(connection):
    """sort key for sorting a connection dictionary

    Args:
        connection of the form '{instancename}:{portname}'
    """
    part1, part2 = connection
    name1, _ = part1.split(":")
    name2, _ = part2.split(":")
    return (min(name1, name2), max(name1, name2))


def _interconnect_ports(block_diag, current_ports, p1, p2):
    """interconnect two ports in a given model

    Args:
        model: the component for which to interconnect the given ports
        p1: the first port name to connect
        p2: the second port name to connect

    Returns:
        the resulting interconnected component, i.e. a component with two ports
        less than the original component.

    Note:
        The interconnect algorithm is based on equation 6 in the paper below::

          Filipsson, Gunnar. "A new general computer algorithm for S-matrix calculation
          of interconnected multiports." 11th European Microwave Conference. IEEE, 1981.
    """
    current_block_diag = {}
    for i in current_ports:
        for j in current_ports:
            vij = _calculate_interconnected_value(
                vij=block_diag.get((i, j), 0.0),
                vik=block_diag.get((i, p1), 0.0),
                vil=block_diag.get((i, p2), 0.0),
                vkj=block_diag.get((p1, j), 0.0),
                vkk=block_diag.get((p1, p1), 0.0),
                vkl=block_diag.get((p1, p2), 0.0),
                vlj=block_diag.get((p2, j), 0.0),
                vlk=block_diag.get((p2, p1), 0.0),
                vll=block_diag.get((p2, p2), 0.0),
            )
            current_block_diag[i, j] = vij
    return current_block_diag


@jax.jit
def _calculate_interconnected_value(vij, vik, vil, vkj, vkk, vkl, vlj, vlk, vll):
    """Calculate an interconnected S-parameter value

    Note:
        The interconnect algorithm is based on equation 6 in the paper below::

          Filipsson, Gunnar. "A new general computer algorithm for S-matrix calculation
          of interconnected multiports." 11th European Microwave Conference. IEEE, 1981.
    """
    result = vij + (
        vkj * vil * (1 - vlk)
        + vlj * vik * (1 - vkl)
        + vkj * vll * vik
        + vlj * vkk * vil
    ) / ((1 - vkl) * (1 - vlk) - vkk * vll)
    return result


def validate_circuit_args(
    instances: Dict[str, Model], connections: Dict[str, str], ports: Dict[str, str]
) -> Tuple[Dict[str, Model], Dict[str, str], Dict[str, str]]:
    """validate the netlist parameters of a circuit

    Args:
        instances: a dictionary with as keys the model names and values
            the model dictionaries.
        connections: a dictionary where both keys and values are strings of the
            form "instancename:portname"
        ports: a dictionary mapping output portnames of the circuit to
            portnames of the form "instancename:portname".

    Returns:
        the validated and possibly slightly modified instances, connections and
        ports dictionaries.
    """

    for model in instances.values():
        validate_model(model)

    if not isinstance(connections, dict):
        msg = (
            "Connections should be a str:str dict "
            "or a list of length-2 tuples. "
            f"Got {connections!r} type {type(connections)}"
        )
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
            msg = f"Connection ports should all be strings. Got: '{port}'"
            assert isinstance(port, str), msg
            msg = f"Connection ports should have format 'modelname:port'. Got: '{port}'"
            assert len(port.split(":")) == 2, msg
            name, _ = port.split(":")
            msg = f"Model '{name}' used in connection "
            msg += f"'{connection[0]}':'{connection[1]}', "
            msg += f"but '{name}' not found in instances dictionary."
            assert name in instances, msg
            msg = f"Duplicate port found in connections: '{port}'"
            assert port not in connection_ports, msg
            connection_ports.add(port)

    output_ports = set()
    for output_port, port in ports.items():
        msg = f"Ports keys in 'ports' should all be strings. Got: '{port}'"
        assert isinstance(port, str), msg
        msg = f"Port values in 'ports' should all be strings. Got: '{output_port}'"
        assert isinstance(output_port, str), msg
        msg = f"Port values in 'ports' should have format 'model:port'. Got: '{port}'"
        assert len(port.split(":")) == 2, msg
        msg = f"Port keys in 'ports' shouldn't contain a ':'. Got: '{output_port}'"
        assert ":" not in output_port, msg
        msg = f"Duplicate port found in ports or connections: '{port}'"
        assert port not in connection_ports, msg
        name, _ = port.split(":")
        msg = f"Model '{name}' used in output port "
        msg += f"'{port}':'{output_port}', "
        msg += f"but '{name}' not found in instances dictionary."
        assert name in instances, msg
        connection_ports.add(port)
        msg = f"Duplicate port found in output ports: '{output_port}'"
        assert output_port not in output_ports, msg
        output_ports.add(output_port)

    return instances, connections, ports
