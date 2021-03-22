""" Common datastructure types used in SAX """

from __future__ import annotations

from textwrap import indent

import numpy as np
import jax.numpy as jnp

from typing import Any, NamedTuple, Dict, Union, Tuple, Callable

__all__ = [
    "Array",
    "Float",
    "ComplexFloat",
    "ModelParams",
    "ModelFunc",
    "ModelDict",
    "Model",
]

Array = Union[jnp.ndarray, np.ndarray]

Float = Union[float, Array]
""" a ComplexFloat type containing floats and float arrays """

ComplexFloat = Union[complex, Float]
""" a ComplexFloat type containing complex floats and complex float arrays """

ModelParams = Union[Dict[str, "ModelParams"], Dict[str, ComplexFloat]]
""" a Parameter Dictionary type """

ModelFunc = Union[
    Callable[[ModelParams], ComplexFloat],
    Callable[[Dict[str, ComplexFloat]], ComplexFloat],
]
""" a Model Function type which takes a single Parameter Dictionary as
arguments and returns a complex ComplexFloat """

ModelDict = Dict[Tuple[str, str], ModelFunc]
""" a Port Function Dictionary is a mapping from port tuples to model functions """


class Model(NamedTuple):
    """a Model tuple

    Args:
        funcs: a dictionary with port combinations (tuples) pointing to Model Functions
        params: a parameter dictionary
    """

    funcs: ModelDict
    params: ModelParams

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"
        s += indent(f"funcs={self._repr_dict(self.funcs)}", "    ")
        s += ",\n"
        s += indent(f"params={self._repr_dict(self.params)}", "    ")
        s += ",\n)"
        return s

    @staticmethod
    def _repr_dict(params) -> str:
        s = "{\n"
        for k, v in params.items():
            if isinstance(v, dict):
                s += indent(f"{repr(k)}: {Model._repr_dict(v)}", "    ") + ",\n"
            else:
                s += f"    {repr(k)}: {v},\n"
        s += "}"
        return s


def is_float(x: Any) -> bool:
    """ check if an object is a float or a float array """
    if isinstance(x, float):
        return True
    if isinstance(x, np.ndarray):
        return x.dtype in (np.float16, np.float32, np.float64, np.float128)
    if isinstance(x, jnp.ndarray):
        return x.dtype in (jnp.float16, jnp.float32, jnp.float64)
    return False


def is_complex(x: Any) -> bool:
    """ check if an object is a complex or a complex array """
    if isinstance(x, complex):
        return True
    if isinstance(x, np.ndarray):
        return x.dtype in (np.complex64, np.complex128)
    if isinstance(x, jnp.ndarray):
        return x.dtype in (jnp.complex64, jnp.complex128)
    return False


def is_complex_float(x: Any) -> bool:
    """ check if an object is a complex float or a complex float array """
    return is_float(x) or is_complex(x)
