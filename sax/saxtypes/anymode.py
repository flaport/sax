"""SAX Types and type coercions.

Numpy type reference: https://numpy.org/doc/stable/reference/arrays.scalars.html
"""

from __future__ import annotations

__all__ = [
    "Model",
    "ModelFactory",
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

from typing import TypeAlias

from .multimode import (
    ModelFactoryMM,
    ModelMM,
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
from .singlemode import (
    ModelFactorySM,
    ModelSM,
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
