"""SAX Backends."""

from __future__ import annotations

import warnings
from typing import Any, cast

from ..saxtypes import Backend, Model, SType
from ..saxtypes.netlist import Instance
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

__all__ = [
    "analyze_circuit",
    "analyze_circuit_additive",
    "analyze_circuit_fg",
    "analyze_circuit_forward",
    "analyze_instances",
    "analyze_instances_additive",
    "analyze_instances_fg",
    "analyze_instances_forward",
    "backend_map",
    "circuit_backends",
    "evaluate_circuit",
    "evaluate_circuit_additive",
    "evaluate_circuit_fg",
    "evaluate_circuit_forward",
    "validate_circuit_backend",
]

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
    instances: dict[str, Instance],
    models: dict[str, Model],
) -> Any:  # noqa: ANN401
    """Analyze circuit instances for the default backend.

    Prepares circuit instances for analysis by the selected backend. This is the
    first step in circuit evaluation, where individual component models are
    prepared for connection analysis.

    Args:
        instances: Dictionary mapping instance names to instance definitions.
        models: Dictionary mapping component names to their model functions.

    Returns:
        Backend-specific analyzed instances data structure.

    Example:
        ```python
        instances = {"wg1": {"component": "waveguide", "settings": {}}}
        models = {"waveguide": my_waveguide_model}
        analyzed = analyze_instances(instances, models)
        ```
    """
    return circuit_backends["default"][0](instances, models)


def analyze_circuit(
    analyzed_instances: Any,  # noqa: ANN401
    connections: dict[str, str],
    ports: dict[str, str],
) -> Any:  # noqa: ANN401
    """Analyze circuit connections for the default backend.

    Analyzes how circuit components are connected together based on the netlist
    connections and ports. This creates the mathematical structure needed for
    circuit evaluation.

    Args:
        analyzed_instances: Output from analyze_instances function.
        connections: Dictionary mapping instance ports to each other.
        ports: Dictionary mapping external port names to instance ports.

    Returns:
        Backend-specific analyzed circuit data structure.

    Example:
        ```python
        connections = {"wg1,out": "wg2,in"}
        ports = {"in": "wg1,in", "out": "wg2,out"}
        analyzed_circuit = analyze_circuit(analyzed_instances, connections, ports)
        ```
    """
    return circuit_backends["default"][1](analyzed_instances, connections, ports)


def evaluate_circuit(
    analyzed: Any,  # noqa: ANN401
    instances: dict[str, SType],
) -> SType:
    """Evaluate circuit S-matrix for the default backend.

    Computes the overall circuit S-matrix from the analyzed circuit structure
    and the evaluated S-matrices of individual instances.

    Args:
        analyzed: Output from analyze_circuit function.
        instances: Dictionary mapping instance names to their evaluated S-matrices.

    Returns:
        Overall circuit S-matrix.

    Example:
        ```python
        # Evaluate individual instances
        s_matrices = {"wg1": wg1_model(wl=1.55), "wg2": wg2_model(wl=1.55)}
        # Compute circuit S-matrix
        circuit_s = evaluate_circuit(analyzed_circuit, s_matrices)
        ```
    """
    return circuit_backends["default"][2](analyzed, instances)


def validate_circuit_backend(backend: str) -> Backend:
    """Validate and normalize a circuit backend name.

    Checks if the specified backend is available and returns the canonical
    backend name. Handles backend aliases and validates availability.

    Args:
        backend: Backend name to validate (case-insensitive).

    Returns:
        Canonical backend name.

    Raises:
        KeyError: If the backend is not available.

    Example:
        ```python
        backend = validate_circuit_backend("klu")
        # Returns "klu" if available

        backend = validate_circuit_backend("default")
        # Returns the default backend (usually "klu" or "filipsson_gunnar")
        ```
    """
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
    return cast(Backend, backend)
