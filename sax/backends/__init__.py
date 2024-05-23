""" SAX Backends """

from __future__ import annotations

import warnings
from typing import Any, Dict

from ..netlist import Component
from ..saxtypes import Model, SType
from .additive import (
    analyze_circuit_additive,
    analyze_instances_additive,
    evaluate_circuit_additive,
)
from .filipsson_gunnar import (
    analyze_circuit_fg,
    analyze_instances_fg,
    evaluate_circuit_fg,
)

circuit_backends = {
    "fg": (
        analyze_instances_fg,
        analyze_circuit_fg,
        evaluate_circuit_fg,
    ),
    "filipsson_gunnar": (
        analyze_instances_fg,
        analyze_circuit_fg,
        evaluate_circuit_fg,
    ),
    "additive": (
        analyze_instances_additive,
        analyze_circuit_additive,
        evaluate_circuit_additive,
    ),
}

try:
    from .klu import analyze_circuit_klu, analyze_instances_klu, evaluate_circuit_klu

    circuit_backends["klu"] = (
        analyze_instances_klu,
        analyze_circuit_klu,
        evaluate_circuit_klu,
    )
    circuit_backends["default"] = (
        analyze_instances_klu,
        analyze_circuit_klu,
        evaluate_circuit_klu,
    )
except ImportError:
    circuit_backends["default"] = (
        analyze_instances_fg,
        analyze_circuit_fg,
        evaluate_circuit_fg,
    )
    warnings.warn(
        "klujax not found. Please install klujax for "
        "better performance during circuit evaluation!"
    )

try:
    from .cuda import (
        analyze_circuit_cuda,
        analyze_instances_cuda,
        evaluate_circuit_cuda,
    )

    circuit_backends["cuda"] = (
        analyze_instances_cuda,
        analyze_circuit_cuda,
        evaluate_circuit_cuda,
    )
    circuit_backends["default"] = (
        analyze_instances_cuda,
        analyze_circuit_cuda,
        evaluate_circuit_cuda,
    )
except ImportError:
    default_backend = "klu" if "klu" in circuit_backends else "fg"
    circuit_backends["default"] = circuit_backends[default_backend]
    warnings.warn(
        "cupy not found. Please install cupy for "
        "better performance during circuit evaluation!"
    )


def analyze_instances(
    instances: Dict[str, Component],
    models: Dict[str, Model],
) -> Any:
    return circuit_backends["default"][0](instances, models)


def analyze_circuit(
    analyzed_instances: Any,
    connections: Dict[str, str],
    ports: Dict[str, str],
) -> Any:
    return circuit_backends["default"][1](analyzed_instances, connections, ports)


def evaluate_circuit(
    analyzed: Any,
    instances: Dict[str, SType],
) -> SType:
    return circuit_backends["default"][2](analyzed, instances)
