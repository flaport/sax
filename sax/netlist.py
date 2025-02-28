def netlist(
    netlist: Any,
    with_unconnected_instances: bool = True,
    with_placements=True,
) -> RecursiveNetlist:
    """Return a netlist from a given dictionary."""
    if isinstance(netlist, RecursiveNetlist):
        net = netlist
    elif isinstance(netlist, Netlist):
        net = RecursiveNetlist(root={"top_level": netlist})
    elif isinstance(netlist, dict):
        if is_recursive(netlist):
            net = RecursiveNetlist.model_validate(netlist)
        else:
            flat_net = Netlist.model_validate(netlist)
            net = RecursiveNetlist.model_validate({"top_level": flat_net})
    else:
        msg = (
            "Invalid argument for `netlist`. "
            "Expected type: dict | Netlist | RecursiveNetlist. "
            f"Got: {type(netlist)}."
        )
        raise ValueError(
            msg,
        )
    if not with_unconnected_instances:
        recnet_dict: RecursiveNetlistDict = _remove_unused_instances(net.model_dump())
        net = RecursiveNetlist.model_validate(recnet_dict)
    if not with_placements:
        for _net in net.root.values():
            _net.placements = {}
    return net


def flatten_netlist(recnet: RecursiveNetlistDict, sep: str = "~"):
    first_name = next(iter(recnet.keys()))
    net = _copy_netlist(recnet[first_name])
    _flatten_netlist(recnet, net, sep)
    return net


@lru_cache
def load_netlist(pic_path: str) -> Netlist:
    with open(pic_path) as file:
        net = yaml.safe_load(file.read())
    return Netlist.model_validate(net)


@lru_cache
def load_recursive_netlist(pic_path: str, ext: str = ".yml"):
    folder_path = os.path.dirname(os.path.abspath(pic_path))

    def _clean_string(path: str) -> str:
        return clean_string(re.sub(ext, "", os.path.split(path)[-1]))

    # the circuit we're interested in should come first:
    netlists: dict[str, Netlist] = {_clean_string(pic_path): Netlist()}

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if not os.path.isfile(path) or not path.endswith(ext):
            continue
        netlists[_clean_string(path)] = load_netlist(path)

    return RecursiveNetlist.model_validate(netlists)


def is_recursive(netlist: AnyNetlist):
    if isinstance(netlist, RecursiveNetlist):
        return True
    if isinstance(netlist, dict):
        return "instances" not in netlist
    return False


def is_not_recursive(netlist: AnyNetlist) -> bool:
    return not is_recursive(netlist)


def get_netlist_instances_by_prefix(
    recursive_netlist: RecursiveNetlist,
    prefix: str,
):
    """Returns a list of all instances with a given prefix in a recursive netlist.

    Args:
        recursive_netlist: The recursive netlist to search.
        prefix: The prefix to search for.

    Returns:
        A list of all instances with the given prefix.
    """
    recursive_netlist_root = recursive_netlist.model_dump()
    result = []
    for key in recursive_netlist_root:
        if key.startswith(prefix):
            result.append(key)
    return result


def get_component_instances(
    recursive_netlist: RecursiveNetlist,
    top_level_prefix: str,
    component_name_prefix: str,
):
    """Returns a dictionary of all instances of a given component in a recursive netlist.

    Args:
        recursive_netlist: The recursive netlist to search.
        top_level_prefix: The prefix of the top level instance.
        component_name_prefix: The name of the component to search for.

    Returns:
        A dictionary of all instances of the given component.
    """
    instance_names = []
    recursive_netlist_root = recursive_netlist.model_dump()

    # Should only be one in a netlist-to-digraph. Can always be very specified.
    top_level_prefixes = get_netlist_instances_by_prefix(
        recursive_netlist,
        prefix=top_level_prefix,
    )
    top_level_prefix = top_level_prefixes[0]
    for key in recursive_netlist_root[top_level_prefix]["instances"]:
        if recursive_netlist_root[top_level_prefix]["instances"][key][
            "component"
        ].startswith(component_name_prefix):
            # Note priority encoding on match.
            instance_names.append(key)
    return {component_name_prefix: instance_names}


def _remove_unused_instances(recursive_netlist: RecursiveNetlistDict):
    recursive_netlist = {**recursive_netlist}

    for name, flat_netlist in recursive_netlist.items():
        recursive_netlist[name] = _remove_unused_instances_flat(flat_netlist)

    return recursive_netlist


def _get_connectivity_netlist(netlist):
    return {
        "instances": natsorted(netlist["instances"]),
        "connections": [
            (c1.split(",")[0], c2.split(",")[0])
            for c1, c2 in netlist["connections"].items()
        ],
        "ports": [(p, c.split(",")[0]) for p, c in netlist["ports"].items()],
    }


def _get_connectivity_graph(netlist):
    graph = nx.Graph()
    connectivity_netlist = _get_connectivity_netlist(netlist)
    for name in connectivity_netlist["instances"]:
        graph.add_node(name)
    for c1, c2 in connectivity_netlist["connections"]:
        graph.add_edge(c1, c2)
    for c1, c2 in connectivity_netlist["ports"]:
        graph.add_edge(c1, c2)
    return graph


def _get_nodes_to_remove(graph, netlist):
    nodes = set()
    for port in netlist["ports"]:
        nodes |= nx.descendants(graph, port)
    nodes_to_remove = set(graph.nodes) - nodes
    return list(nodes_to_remove)


def _remove_unused_instances_flat(flat_netlist: NetlistDict) -> NetlistDict:
    flat_netlist = {**flat_netlist}

    flat_netlist["instances"] = {**flat_netlist["instances"]}
    flat_netlist["connections"] = {**flat_netlist["connections"]}
    flat_netlist["ports"] = {**flat_netlist["ports"]}

    graph = _get_connectivity_graph(flat_netlist)
    names = set(_get_nodes_to_remove(graph, flat_netlist))

    for name in list(names):
        if name in flat_netlist["instances"]:
            del flat_netlist["instances"][name]

    for conn1, conn2 in list(flat_netlist["connections"].items()):
        for conn in [conn1, conn2]:
            name, _ = conn.split(",")
            if name in names and conn1 in flat_netlist["connections"]:
                del flat_netlist["connections"][conn1]

    for pname, conn in list(flat_netlist["ports"].items()):
        name, _ = conn.split(",")
        if name in names and pname in flat_netlist["ports"]:
            del flat_netlist["ports"][pname]

    return flat_netlist


def _copy_netlist(net):
    return {
        k: deepcopy(v)
        for k, v in net.items()
        if k in ["instances", "connections", "ports"]
    }


def _flatten_netlist(recnet, net, sep) -> None:
    for name, instance in list(net["instances"].items()):
        component = instance["component"]
        if component not in recnet:
            continue
        del net["instances"][name]
        child_net = _copy_netlist(recnet[component])
        _flatten_netlist(recnet, child_net, sep)
        for iname, iinstance in child_net["instances"].items():
            net["instances"][f"{name}{sep}{iname}"] = iinstance
        ports = {k: f"{name}{sep}{v}" for k, v in child_net["ports"].items()}
        for ip1, ip2 in list(net["connections"].items()):
            n1, p1 = ip1.split(",")
            n2, p2 = ip2.split(",")
            if n1 == name:
                del net["connections"][ip1]
                if p1 not in ports:
                    warnings.warn(
                        f"Port {ip1} not found. Connection {ip1}<->{ip2} ignored.",
                        stacklevel=2,
                    )
                    continue
                net["connections"][ports[p1]] = ip2
            elif n2 == name:
                if p2 not in ports:
                    warnings.warn(
                        f"Port {ip2} not found. Connection {ip1}<->{ip2} ignored.",
                        stacklevel=2,
                    )
                    continue
                net["connections"][ip1] = ports[p2]
        for ip1, ip2 in child_net["connections"].items():
            net["connections"][f"{name}{sep}{ip1}"] = f"{name}{sep}{ip2}"
        for p, ip2 in list(net["ports"].items()):
            try:
                n2, p2 = ip2.split(",")
            except ValueError:
                warnings.warn(
                    f"Unconventional port definition ignored: {p}->{ip2}.", stacklevel=2
                )
                continue
            if n2 == name:
                if p2 in ports:
                    net["ports"][p] = ports[p2]
                else:
                    del net["ports"][p]


@overload
def rename_instances(netlist: Netlist, mapping: dict[str, str]) -> Netlist: ...


@overload
def rename_instances(
    netlist: RecursiveNetlist,
    mapping: dict[str, str],
) -> RecursiveNetlist: ...


@overload
def rename_instances(netlist: NetlistDict, mapping: dict[str, str]) -> NetlistDict: ...


@overload
def rename_instances(
    netlist: RecursiveNetlistDict,
    mapping: dict[str, str],
) -> RecursiveNetlistDict: ...


def rename_instances(
    netlist: Netlist | RecursiveNetlist | NetlistDict | RecursiveNetlistDict,
    mapping: dict[str, str],
) -> Netlist | RecursiveNetlist | NetlistDict | RecursiveNetlistDict:
    given_as_dict = isinstance(netlist, dict)

    if is_recursive(netlist):
        netlist = RecursiveNetlist.model_validate(netlist)
    else:
        netlist = Netlist.model_validate(netlist)

    if isinstance(netlist, RecursiveNetlist):
        net = RecursiveNetlist(
            **{
                k: rename_instances(v, mapping).model_dump()
                for k, v in netlist.root.items()
            },
        )
        return net if not given_as_dict else net.model_dump()

    # it's a sax.Netlist now:
    inverse_mapping = {v: k for k, v in mapping.items()}
    if len(inverse_mapping) != len(mapping):
        msg = "Duplicate names to map onto found."
        raise ValueError(msg)
    instances = {mapping.get(k, k): v for k, v in netlist.instances.items()}
    connections = {}
    for ip1, ip2 in netlist.connections.items():
        i1, p1 = ip1.split(",")
        i2, p2 = ip2.split(",")
        i1 = mapping.get(i1, i1)
        i2 = mapping.get(i2, i2)
        connections[f"{i1},{p1}"] = f"{i2},{p2}"
    ports = {}
    for q, ip in netlist.ports.items():
        i, p = ip.split(",")
        i = mapping.get(i, i)
        ports[q] = f"{i},{p}"

    placements = {mapping.get(k, k): v for k, v in netlist.placements.items()}
    net = Netlist(
        instances=instances,
        connections=connections,
        ports=ports,
        placements=placements,
        settings=netlist.settings,
    )
    return net if not given_as_dict else net.model_dump()


@overload
def rename_models(netlist: Netlist, mapping: dict[str, str]) -> Netlist: ...


@overload
def rename_models(
    netlist: RecursiveNetlist,
    mapping: dict[str, str],
) -> RecursiveNetlist: ...


@overload
def rename_models(netlist: NetlistDict, mapping: dict[str, str]) -> NetlistDict: ...


@overload
def rename_models(
    netlist: RecursiveNetlistDict,
    mapping: dict[str, str],
) -> RecursiveNetlistDict: ...


def rename_models(
    netlist: Netlist | RecursiveNetlist | NetlistDict | RecursiveNetlistDict,
    mapping: dict[str, str],
) -> Netlist | RecursiveNetlist | NetlistDict | RecursiveNetlistDict:
    given_as_dict = isinstance(netlist, dict)

    if is_recursive(netlist):
        netlist = RecursiveNetlist.model_validate(netlist)
    else:
        netlist = Netlist.model_validate(netlist)

    if isinstance(netlist, RecursiveNetlist):
        net = RecursiveNetlist(
            **{
                k: rename_models(v, mapping).model_dump()
                for k, v in netlist.root.items()
            },
        )
        return net if not given_as_dict else net.model_dump()

    # it's a sax.Netlist now:
    inverse_mapping = {v: k for k, v in mapping.items()}
    if len(inverse_mapping) != len(mapping):
        msg = "Duplicate names to map onto found."
        raise ValueError(msg)

    instances = {}
    for k, instance in netlist.instances.items():
        given_as_str = False
        if isinstance(instance, str):
            given_as_str = True
            instance = {
                "component": instance,
                "settings": {},
            }
        elif isinstance(instance, Component):
            instance = instance.model_dump()
        assert isinstance(instance, dict)
        instance["component"] = mapping.get(
            instance["component"],
            instance["component"],
        )
        if given_as_str:
            instances[k] = instance["component"]
        else:
            instances[k] = instance

    net = Netlist(
        instances=instances,
        connections=netlist.connections,
        ports=netlist.ports,
        placements=netlist.placements,
        settings=netlist.settings,
    )
    return net if not given_as_dict else net.model_dump()


def _nets_to_connections(nets: list[dict], connections: dict):
    connections = dict(connections.items())
    inverse_connections = {v: k for k, v in connections.items()}

    def _is_connected(p):
        return (p in connections) or (p in inverse_connections)

    def _add_connection(p, q) -> None:
        connections[p] = q
        inverse_connections[q] = p

    def _get_connected_port(p):
        if p in connections:
            return connections[p]
        return inverse_connections[p]

    for net in nets:
        p = net["p1"]
        q = net["p2"]
        if _is_connected(p):
            _q = _get_connected_port(p)
            msg = (
                "SAX currently does not support multiply connected ports. "
                f"Got {p}<->{q} and {p}<->{_q}"
            )
            raise ValueError(
                msg,
            )
        if _is_connected(q):
            _p = _get_connected_port(q)
            msg = (
                "SAX currently does not support multiply connected ports. "
                f"Got {p}<->{q} and {_p}<->{q}"
            )
            raise ValueError(
                msg,
            )
        _add_connection(p, q)
    return connections
