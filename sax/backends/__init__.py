""" SAX Backends """

from __future__ import annotations

import warnings
from typing import Any, Dict

from .additive import analyze_circuit_additive, evaluate_circuit_additive
from .filipsson_gunnar import analyze_circuit_fg, evaluate_circuit_fg
from ..saxtypes import SType

circuit_backends = {
    "fg": (analyze_circuit_fg, evaluate_circuit_fg),
    "filipsson_gunnar": (analyze_circuit_fg, evaluate_circuit_fg),
    "additive": (analyze_circuit_additive, evaluate_circuit_additive),
}

try:
    from .klu import analyze_circuit_klu, evaluate_circuit_klu
    circuit_backends["klu"] = (analyze_circuit_klu, evaluate_circuit_klu)
    circuit_backends["default"] = (analyze_circuit_klu, evaluate_circuit_klu)
except ImportError:
    circuit_backends["default"] = (analyze_circuit_fg, evaluate_circuit_fg)
    warnings.warn(
        "klujax not found. Please install klujax for "
        "better performance during circuit evaluation!"
    )


def analyze_circuit(connections: Dict[str, str], ports: Dict[str, str]) -> Any:
    return circuit_backends["default"][0](connections, ports)


def evaluate_circuit(analyzed: Any, instances: Dict[str, SType]) -> SType:
    return circuit_backends["default"][1](analyzed, instances)
