"""SAX Backends."""

from __future__ import annotations

import warnings
from typing import Any, Dict

import sax

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
from .forward_only import (
    analyze_circuit_forward,
    analyze_instances_forward,
    evaluate_circuit_forward,
)

circuit_backends = {
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
    "forward": (
        analyze_instances_forward,
        analyze_circuit_forward,
        evaluate_circuit_forward,
    ),
}

backend_map = {
    "fg": "filipsson_gunnar",
    "filipsson_gunnar": "filipsson_gunnar",
    "additive": "additive",
    "forward": "forward",
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
    backend_map["klu"] = "klu"
    backend_map["default"] = "klu"
except ImportError:
    circuit_backends["default"] = (
        analyze_instances_fg,
        analyze_circuit_fg,
        evaluate_circuit_fg,
    )
    backend_map["default"] = "filipsson_gunnar"
    warnings.warn(
        "klujax not found. Please install klujax for "
        "better performance during circuit evaluation!",
        stacklevel=2,
    )


def analyze_instances(
    instances: dict[str, Component],
    models: dict[str, Model],
) -> Any:
    return circuit_backends["default"][0](instances, models)


def analyze_circuit(
    analyzed_instances: Any,
    connections: dict[str, str],
    ports: dict[str, str],
) -> Any:
    return circuit_backends["default"][1](analyzed_instances, connections, ports)


def evaluate_circuit(
    analyzed: Any,
    instances: dict[str, SType],
) -> SType:
    return circuit_backends["default"][2](analyzed, instances)


def validate_circuit_backend(backend: str) -> sax.Backend:
    backend = backend.lower()
    backend = backend_map.get(backend, backend)
    # assert valid circuit_backend
    if backend not in circuit_backends:
        msg = (
            f"circuit backend {backend} not found. Allowed circuit backends: "
            f"{', '.join(circuit_backends.keys())}."
        )
        raise KeyError(
            msg,
        )
    return backend
