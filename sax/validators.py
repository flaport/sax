"""Validate stuff."""

import inspect
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import jax.numpy as jnp
from numpy.exceptions import ComplexWarning

from .saxtypes import Settings, SType, is_mixedmode


def try_complex_float(f: Any) -> Any:
    """Try converting an object to float, return unchanged object on fail."""
    with warnings.catch_warnings():
        warnings.filterwarnings(action="error", category=ComplexWarning)
        try:
            return jnp.asarray(f, dtype=float)
        except ComplexWarning:
            return jnp.asarray(f, dtype=complex)
        except (ValueError, TypeError):
            pass
    return f


def validate_settings(settings: Settings) -> Settings:
    """Validate a parameter dictionary."""
    _settings = {}
    for k, v in settings.items():
        if isinstance(v, dict):
            _settings[k] = validate_settings(v)
        else:
            _settings[k] = try_complex_float(v)
    return _settings


def validate_not_mixedmode(S: SType) -> None:
    """Validate that an stype is not 'mixed mode' (i.e. invalid).

    Args:
        S: the stype to validate
    """
    if is_mixedmode(S):  # mixed mode
        msg = (
            "Given SType is neither multimode or singlemode. Please check the port "
            "names: they should either ALL contain the '@' separator (multimode) "
            "or NONE should contain the '@' separator (singlemode)."
        )
        raise ValueError(
            msg,
        )


def validate_multimode(S: SType, modes: Iterable[str] = ("te", "tm")) -> None:
    """Validate that an stype is multimode and that the given modes are present."""
    from .utils import get_ports

    try:
        current_modes = {p.split("@")[1] for p in get_ports(S)}
    except IndexError as e:
        msg = "The given SType is not multimode."
        raise TypeError(msg) from e
    for mode in modes:
        if mode not in current_modes:
            msg = f"Could not find mode '{mode}' in one of the multimode models."
            raise ValueError(msg)


def validate_sdict(sdict: Any) -> None:
    """Validate an `SDict`."""
    if not isinstance(sdict, dict):
        msg = "An SDict should be a dictionary."
        raise TypeError(msg)
    for ports in sdict:
        if not isinstance(ports, tuple) and len(ports) != 2:
            msg = f"SDict keys should be length-2 tuples. Got {ports}"
            raise TypeError(msg)
        p1, p2 = ports
        if not isinstance(p1, str) or not isinstance(p2, str):
            msg = (
                f"SDict ports should be strings. Got {ports} "
                f"({type(ports[0])}, {type(ports[1])})"
            )
            raise TypeError(msg)


def validate_model(model: Callable) -> None:
    """Validate the parameters of a model."""
    positional_arguments = []
    for param in inspect.signature(model).parameters.values():
        if param.default is inspect.Parameter.empty:
            positional_arguments.append(param.name)  # noqa: PERF401
    if positional_arguments:
        msg = (
            f"model '{model}' takes positional "
            f"arguments {', '.join(positional_arguments)} "
            "and hence is not a valid SAX Model! "
            "A SAX model should ONLY take keyword arguments (or no arguments at all)."
        )
        raise ValueError(msg)
