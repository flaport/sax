""" Common datastructure types used in SAX """

from __future__ import annotations

from typing import Optional, Dict, Union, Tuple, Callable, Any

import numpy as np
import jax.numpy as jnp


Float = Union[float, jnp.ndarray, np.ndarray]
""" a Float type containing floats and float arrays """

ComplexFloat = Union[complex, float, jnp.ndarray, np.ndarray]
""" a ComplexFloat type containing complex floats and complex float arrays """

ParamsDict = Dict[str, Union["ParamsDict", Float]]
""" a Parameter Dictionary type """

ModelFunc = Callable[[ParamsDict], ComplexFloat]
""" a Model Function type which takes a single Parameter Dictionary as
arguments and returns a complex Float """

ModelDict = Dict[Union[Tuple[str, str], str], Union[ModelFunc, ParamsDict]]
""" a Model Dictionary type which consists of port combinations (tuples) pointing
to Model Functions and a subdirectorty 'params' which is a Parameter Dictionary. """


def is_float(x: Any) -> bool:
    """ check if an object is a float or a float array """
    if isinstance(x, float):
        return True
    if isinstance(x, np.ndarray):
        return x.dtype in (np.float16, np.float32, np.float64, np.float128)
    if isinstance(x, jnp.ndarray):
        return x.dtype in (jnp.float16, jnp.float32, jnp.float64)
    return False
