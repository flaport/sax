""" SAX Backends """

from __future__ import annotations

import warnings
from typing import Any, Dict

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp

    JAX_AVAILABLE = False

try:
    import klujax

    KLUJAX_AVAILABLE = True
except ImportError:
    KLUJAX_AVAILABLE = False

from .additive import analyze_circuit_additive, evaluate_circuit_additive
from .filipsson_gunnar import analyze_circuit_fg, evaluate_circuit_fg
from ..typing_ import SType, sdict

if JAX_AVAILABLE and KLUJAX_AVAILABLE:
    from .klu import analyze_circuit_klu, evaluate_circuit_klu


circuit_backends = {
    "fg": (analyze_circuit_fg, evaluate_circuit_fg),
    "filipsson_gunnar": (analyze_circuit_fg, evaluate_circuit_fg),
    "additive": (analyze_circuit_additive, evaluate_circuit_additive),
}

if JAX_AVAILABLE and KLUJAX_AVAILABLE:
    circuit_backends["klu"] = (analyze_circuit_klu, evaluate_circuit_klu)
    circuit_backends["default"] = (analyze_circuit_klu, evaluate_circuit_klu)
else:
    circuit_backends["default"] = (analyze_circuit_fg, evaluate_circuit_fg)
    warnings.warn(
        "klujax not found. Please install klujax for better performance during circuit evaluation!"
    )


def analyze_circuit(connections: Dict[str, str], ports: Dict[str, str]) -> Any:
    return circuit_backends["default"][0](connections, ports)


def evaluate_circuit(analyzed: Any, instances: Dict[str, SType]) -> SType:
    return circuit_backends["default"][1](analyzed, instances)
