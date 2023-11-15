""" SAX netlist parsing and utilities """

from __future__ import annotations

import os
import re
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Optional, TypedDict, Union

import black
import networkx as nx
import numpy as np
import yaml
from natsort import natsorted

from .utils import clean_string, hash_dict

try:
    from pydantic.v1 import BaseModel, Extra, Field, ValidationError, validator
except ImportError:
    from pydantic import BaseModel, Extra, Field, ValidationError, validator


def netlist(dic: Dict) -> RecursiveNetlist:
    """return a netlist from a given dictionary"""
    if isinstance(dic, RecursiveNetlist):
        return dic
    elif isinstance(dic, Netlist):
        dic = dic.dict()
    try:
        flat_net = Netlist.parse_obj(dic)
        net = RecursiveNetlist.parse_obj({"top_level": flat_net})
    except ValidationError:
        net = RecursiveNetlist.parse_obj(dic)
    return net


class _BaseModel(BaseModel):  # type: ignore
    class Config:
        extra = Extra.ignore
        allow_mutation = False
        frozen = True
        json_encoders = {np.ndarray: lambda arr: np.round(arr, 12).tolist()}

    def __repr__(self):
        s = super().__repr__()
        s = black.format_str(s, mode=black.Mode())
        return s

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash_dict(self.dict())


class Component(_BaseModel):
    class Config:
        extra = Extra.ignore
        allow_mutation = False
        frozen = True
        json_encoders = {np.ndarray: lambda arr: np.round(arr, 12).tolist()}

    component: Union[str, Dict[str, Any]] = Field(..., title="Component")
    settings: Optional[Dict[str, Any]] = Field(None, title="Settings")

    # this was added:
    @validator("component")
    def validate_component_name(cls, value):
        if "," in value:
            raise ValueError(
                f"Invalid component string. Should not contain ','. Got: {value}"
            )
        value = value.split("$")[0]
        return clean_string(value)


class PortEnum(Enum):
    ce = "ce"
    cw = "cw"
    nc = "nc"
    ne = "ne"
    nw = "nw"
    sc = "sc"
    se = "se"
    sw = "sw"
    center = "center"
    cc = "cc"


class Placement(_BaseModel):
    class Config:
        extra = Extra.ignore
        allow_mutation = False
        frozen = True
        json_encoders = {np.ndarray: lambda arr: np.round(arr, 12).tolist()}

    x: Optional[Union[str, float]] = Field(0, title="X")
    y: Optional[Union[str, float]] = Field(0, title="Y")
    xmin: Optional[Union[str, float]] = Field(None, title="Xmin")
    ymin: Optional[Union[str, float]] = Field(None, title="Ymin")
    xmax: Optional[Union[str, float]] = Field(None, title="Xmax")
    ymax: Optional[Union[str, float]] = Field(None, title="Ymax")
    dx: Optional[float] = Field(0, title="Dx")
    dy: Optional[float] = Field(0, title="Dy")
    port: Optional[Union[str, PortEnum]] = Field(None, title="Port")
    rotation: Optional[int] = Field(0, title="Rotation")
    mirror: Optional[bool] = Field(False, title="Mirror")


class Route(_BaseModel):
    class Config:
        extra = Extra.ignore
        allow_mutation = False
        frozen = True
        json_encoders = {np.ndarray: lambda arr: np.round(arr, 12).tolist()}

    links: Dict[str, str] = Field(..., title="Links")
    settings: Optional[Dict[str, Any]] = Field(None, title="Settings")
    routing_strategy: Optional[str] = Field(None, title="Routing Strategy")


class Netlist(_BaseModel):
    class Config:
        extra = Extra.ignore
        allow_mutation = False
        frozen = True
        json_encoders = {np.ndarray: lambda arr: np.round(arr, 12).tolist()}

    instances: Dict[str, Component] = Field(..., title="Instances")
    connections: Optional[Dict[str, str]] = Field(None, title="Connections")
    ports: Optional[Dict[str, str]] = Field(None, title="Ports")
    placements: Optional[Dict[str, Placement]] = Field(None, title="Placements")

    # these were removed (irrelevant for SAX):

    # routes: Optional[Dict[str, Route]] = Field(None, title='Routes')
    # name: Optional[str] = Field(None, title='Name')
    # info: Optional[Dict[str, Any]] = Field(None, title='Info')
    # settings: Optional[Dict[str, Any]] = Field(None, title='Settings')
    # pdk: Optional[str] = Field(None, title='Pdk')

    # these are extra additions:

    @validator("instances", pre=True)
    def coerce_different_type_instance_into_component_model(cls, instances):
        new_instances = {}
        for k, v in instances.items():
            if isinstance(v, str):
                v = {
                    "component": v,
                    "settings": {},
                }
            new_instances[k] = v

        return new_instances

    @staticmethod
    def clean_instance_string(value):
        if "," in value:
            raise ValueError(
                f"Invalid instance string. Should not contain ','. Got: {value}"
            )
        return clean_string(value)

    @validator("instances")
    def validate_instance_names(cls, instances):
        return {cls.clean_instance_string(k): v for k, v in instances.items()}

    @validator("placements")
    def validate_placement_names(cls, placements):
        if placements is not None:
            return {cls.clean_instance_string(k): v for k, v in placements.items()}
        return {}

    @classmethod
    def clean_connection_string(cls, value):
        *comp, port = value.split(",")
        comp = cls.clean_instance_string(",".join(comp))
        return f"{comp},{port}"

    @validator("connections")
    def validate_connection_names(cls, connections):
        return {
            cls.clean_connection_string(k): cls.clean_connection_string(v)
            for k, v in connections.items()
        }

    @validator("ports")
    def validate_port_names(cls, ports):
        return {
            cls.clean_instance_string(k): cls.clean_connection_string(v)
            for k, v in ports.items()
        }


class RecursiveNetlist(_BaseModel):
    class Config:
        extra = Extra.ignore
        allow_mutation = False
        frozen = True

    __root__: Dict[str, Netlist]


class NetlistDict(TypedDict):
    instances: Dict
    connections: Dict[str, str]
    ports: Dict[str, str]


RecursiveNetlistDict = Dict[str, NetlistDict]


@lru_cache()
def load_netlist(pic_path) -> Netlist:
    with open(pic_path, "r") as file:
        net = yaml.safe_load(file.read())
    return Netlist.parse_obj(net)


@lru_cache()
def load_recursive_netlist(pic_path, ext=".yml"):
    folder_path = os.path.dirname(os.path.abspath(pic_path))

    def _clean_string(path: str) -> str:
        return clean_string(re.sub(ext, "", os.path.split(path)[-1]))

    # the circuit we're interested in should come first:
    netlists: Dict[str, Netlist] = {_clean_string(pic_path): Netlist()}

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if not os.path.isfile(path) or not path.endswith(ext):
            continue
        netlists[_clean_string(path)] = load_netlist(path)

    return RecursiveNetlist.parse_obj(netlists)


def get_netlist_instances_by_prefix(
    recursive_netlist: RecursiveNetlist,
    prefix: str,
):
    """
    Returns a list of all instances with a given prefix in a recursive netlist.

    Args:
        recursive_netlist: The recursive netlist to search.
        prefix: The prefix to search for.

    Returns:
        A list of all instances with the given prefix.
    """
    recursive_netlist_root = recursive_netlist.dict()["__root__"]
    result = []
    for key in recursive_netlist_root.keys():
        if key.startswith(prefix):
            result.append(key)
    return result


def get_component_instances(
    recursive_netlist: RecursiveNetlist,
    top_level_prefix: str,
    component_name_prefix: str,
):
    """
    Returns a dictionary of all instances of a given component in a recursive netlist.

    Args:
        recursive_netlist: The recursive netlist to search.
        top_level_prefix: The prefix of the top level instance.
        component_name_prefix: The name of the component to search for.

    Returns:
        A dictionary of all instances of the given component.
    """
    instance_names = []
    recursive_netlist_root = recursive_netlist.dict()["__root__"]

    # Should only be one in a netlist-to-digraph. Can always be very specified.
    top_level_prefixes = get_netlist_instances_by_prefix(
        recursive_netlist, prefix=top_level_prefix
    )
    top_level_prefix = top_level_prefixes[0]
    for key in recursive_netlist_root[top_level_prefix]["instances"]:
        if recursive_netlist_root[top_level_prefix]["instances"][key][
            "component"
        ].startswith(component_name_prefix):
            # Note priority encoding on match.
            instance_names.append(key)
    return {component_name_prefix: instance_names}


def remove_unused_instances(recursive_netlist: RecursiveNetlistDict):
    recursive_netlist = {**recursive_netlist}

    for name, flat_netlist in recursive_netlist.items():
        recursive_netlist[name] = _remove_unused_instances_flat(flat_netlist)

    return recursive_netlist


def _get_connectivity_netlist(netlist):
    connectivity_netlist = {
        "instances": natsorted(netlist["instances"]),
        "connections": [
            (c1.split(",")[0], c2.split(",")[0])
            for c1, c2 in netlist["connections"].items()
        ],
        "ports": [(p, c.split(",")[0]) for p, c in netlist["ports"].items()],
    }
    return connectivity_netlist


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
