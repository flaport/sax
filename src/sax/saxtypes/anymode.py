"""SAX Types and type coercions.

Numpy type reference: https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

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

from typing import Any, Literal, NamedTuple, TypeAlias

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
ModelFactory: TypeAlias = ModelFactorySM | ModelFactoryMM
PortCombination: TypeAlias = PortCombinationSM | PortCombinationMM
PortMap: TypeAlias = PortMapSM | PortMapMM
SCoo: TypeAlias = SCooSM | SCooMM
SCooModel: TypeAlias = SCooModelSM | SCooModelMM
SCooModelFactory: TypeAlias = SCooModelFactorySM | SCooModelFactoryMM
SDense: TypeAlias = SDenseSM | SDenseMM
SDenseModel: TypeAlias = SDenseModelSM | SDenseModelMM
SDenseModelFactory: TypeAlias = SDenseModelFactorySM | SDenseModelFactoryMM
SDict: TypeAlias = SDictSM | SDictMM
SDictModel: TypeAlias = SDictModelSM | SDictModelMM
SDictModelFactory: TypeAlias = SDictModelFactorySM | SDictModelFactoryMM
SType: TypeAlias = STypeSM | STypeMM
Backend: TypeAlias = Literal["filipsson_gunnar", "additive", "forward", "klu"]
BackendOrDefault: TypeAlias = Backend | Literal["default"]
Models: TypeAlias = ModelsSM | ModelsMM


class CircuitInfo(NamedTuple):
    """Information about the circuit function you created."""

    dag: nx.Graph[Any]
    models: dict[Name, Model]
    backend: Backend
