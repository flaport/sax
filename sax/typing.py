""" Common datastructure types used in SAX """

import numpy as np
import jax.numpy as jnp
from typing import Optional, Dict, Union, Tuple, Callable, Any


Float = Union[float, jnp.ndarray, np.ndarray]

ComplexFloat = Union[complex, float, jnp.ndarray, np.ndarray]

ParamsDict = Dict[str, Union["ParamsDict", Float]]

ModelFunc = Callable[[ParamsDict], ComplexFloat]

ModelDict = Dict[Union[Tuple[str, str], str], Union[ModelFunc, ParamsDict]]


def is_float(x):
    if isinstance(x, float):
        return True
    if isinstance(x, np.ndarray):
        return x.dtype in (np.float16, np.float32, np.float64, np.float128)
    if isinstance(x, jnp.ndarray):
        return x.dtype in (jnp.float16, jnp.float32, jnp.float64)
    return False
