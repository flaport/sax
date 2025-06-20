"""SAX Anymode Types and type coercions.

This module provides unified type aliases that work with both single-mode and
multi-mode optical circuits. It combines the single-mode and multi-mode type
systems into a common interface.

References:
    Numpy type reference: https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

from __future__ import annotations

__all__ = [
    "Backend",
    "BackendOrDefault",
    "CircuitInfo",
    "Model",
    "ModelFactory",
    "Models",
    "PortCombination",
    "PortMap",
    "SCoo",
    "SCooModel",
    "SCooModelFactory",
    "SDense",
    "SDenseModel",
    "SDenseModelFactory",
    "SDict",
    "SDictModel",
    "SDictModelFactory",
    "SType",
]

from typing import Literal, NamedTuple, TypeAlias

import networkx as nx

from sax.saxtypes.core import Name
from sax.saxtypes.multimode import (
    ModelFactoryMM,
    ModelMM,
    ModelsMM,
    PortCombinationMM,
    PortMapMM,
    SCooMM,
    SCooModelFactoryMM,
    SCooModelMM,
    SDenseMM,
    SDenseModelFactoryMM,
    SDenseModelMM,
    SDictMM,
    SDictModelFactoryMM,
    SDictModelMM,
    STypeMM,
)
from sax.saxtypes.singlemode import (
    ModelFactorySM,
    ModelSM,
    ModelsSM,
    PortCombinationSM,
    PortMapSM,
    SCooModelFactorySM,
    SCooModelSM,
    SCooSM,
    SDenseModelFactorySM,
    SDenseModelSM,
    SDenseSM,
    SDictModelFactorySM,
    SDictModelSM,
    SDictSM,
    STypeSM,
)

Model: TypeAlias = ModelSM | ModelMM
"""A SAX model function that works with either single-mode or multi-mode circuits."""

ModelFactory: TypeAlias = ModelFactorySM | ModelFactoryMM
"""A SAX model factory function that works with single-mode or multi-mode circuits."""

PortCombination: TypeAlias = PortCombinationSM | PortCombinationMM
"""A pair of port names, either single-mode or multi-mode format."""

PortMap: TypeAlias = PortMapSM | PortMapMM
"""A mapping from port names to matrix indices, supporting both mode types."""

SCoo: TypeAlias = SCooSM | SCooMM
"""A sparse COO format S-matrix, supporting both single-mode and multi-mode."""

SCooModel: TypeAlias = SCooModelSM | SCooModelMM
"""A model function that produces SCoo S-matrices in either mode format."""

SCooModelFactory: TypeAlias = SCooModelFactorySM | SCooModelFactoryMM
"""A model factory that produces SCoo models in either mode format."""

SDense: TypeAlias = SDenseSM | SDenseMM
"""A dense S-matrix representation, supporting both single-mode and multi-mode."""

SDenseModel: TypeAlias = SDenseModelSM | SDenseModelMM
"""A model function that produces SDense S-matrices in either mode format."""

SDenseModelFactory: TypeAlias = SDenseModelFactorySM | SDenseModelFactoryMM
"""A model factory that produces SDense models in either mode format."""

SDict: TypeAlias = SDictSM | SDictMM
"""A dictionary-based sparse S-matrix, supporting both single-mode and multi-mode."""

SDictModel: TypeAlias = SDictModelSM | SDictModelMM
"""A model function that produces SDict S-matrices in either mode format."""

SDictModelFactory: TypeAlias = SDictModelFactorySM | SDictModelFactoryMM
"""A model factory that produces SDict models in either mode format."""

SType: TypeAlias = STypeSM | STypeMM
"""Any S-matrix type (SDict, SDense, SCoo) in single-mode or multi-mode format."""

Backend: TypeAlias = Literal["filipsson_gunnar", "additive", "forward", "klu"]
"""Available SAX backend algorithms for circuit simulation."""

BackendOrDefault: TypeAlias = Backend | Literal["default"]
"""Backend specification allowing 'default' to use the system default backend."""

Models: TypeAlias = ModelsSM | ModelsMM
"""A collection of model functions, supporting both single-mode and multi-mode."""


class CircuitInfo(NamedTuple):
    """Information about a SAX circuit function.

    This class contains metadata about the circuit structure, models used,
    and backend selected during circuit compilation.

    Attributes:
        dag: The directed acyclic graph representing the circuit topology.
        models: Dictionary mapping instance names to their model functions.
        backend: The backend algorithm used for circuit simulation.

    Examples:
        ```python
        import sax

        # After creating a circuit
        circuit_func, info = sax.circuit(netlist, models, backend="klu")

        # Access circuit information
        print(f"Circuit uses {len(info.models)} models")
        print(f"Backend: {info.backend}")
        print(f"Circuit has {len(info.dag.nodes)} instances")
        ```
    """

    dag: nx.DiGraph
    """The circuit topology as a directed acyclic graph."""

    models: dict[Name, Model]
    """Mapping from instance names to their model functions."""

    backend: Backend
    """The backend algorithm used for circuit simulation."""
