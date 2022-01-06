""" Common datastructure types used in SAX """

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pathlib

from typing import Any, Dict, Union, Tuple, Callable


Array = Union[jnp.ndarray, np.ndarray]

Float = Union[float, Array]
""" a ComplexFloat type containing floats and float arrays """

ComplexFloat = Union[complex, Float]
""" a ComplexFloat type containing complex floats and complex float arrays """

PDict = Union[Dict[str, "PDict"], Dict[str, ComplexFloat]]
""" a Parameter Dictionary type """

SDict = Dict[Tuple[str, str], ComplexFloat]
""" a S-Parameter Dictionary type """

Model = Callable[..., SDict]
""" a Model Function type which takes a single Parameter Dictionary as
arguments and returns an S parameter dictionary """

ModelFactory = Callable[..., Model]
""" A SAX Model Factory is any keyword-only function that returns a sax Model """

PathType = Union[str, pathlib.Path]
Strs = Tuple[str, ...]


def is_float(x: Any) -> bool:
    """check if an object is a float or a float array"""
    if isinstance(x, float):
        return True
    if isinstance(x, np.ndarray):
        return x.dtype in (np.float16, np.float32, np.float64, np.float128)
    if isinstance(x, jnp.ndarray):
        return x.dtype in (jnp.float16, jnp.float32, jnp.float64)
    return False


def is_complex(x: Any) -> bool:
    """check if an object is a complex or a complex array"""
    if isinstance(x, complex):
        return True
    if isinstance(x, np.ndarray):
        return x.dtype in (np.complex64, np.complex128)
    if isinstance(x, jnp.ndarray):
        return x.dtype in (jnp.complex64, jnp.complex128)
    return False


def is_complex_float(x: Any) -> bool:
    """check if an object is a complex float or a complex float array"""
    return is_float(x) or is_complex(x)


__all__ = ["Array", "Float", "ComplexFloat", "PDict", "SDict", "Model"]
