""" Useful functions for working with SAX. """

import pickle
import numpy as np
import jax.numpy as jnp

from .typing import Any, Union, Tuple, Callable, Dict, ParamsDict, ModelDict


def load(name: str) -> object:
    """load an object using pickle

    Args:
        name: the name to load

    Returns:
        the unpickled object.
    """
    with open(name, "rb") as file:
        obj = pickle.load(file)
    return obj


def save(obj: object, name: str):
    """save an object using pickle

    Args:
        obj: the object to save
        name: the name to save the object under
    """
    with open(name, "wb") as file:
        pickle.dump(obj, file)


def validate_params(params: ParamsDict):
    """validate a parameter dictionary

    params: the parameter dictionary. This dictionary should be a possibly
        nested dictionary of floats.
    """
    if not params:
        return

    is_dict_dict = all(isinstance(v, dict) for v in params.values())
    if not is_dict_dict:
        for k, v in params.items():
            msg = f"Wrong parameter dictionary format. Should be a (possibly nested) "
            msg += f"dictionary of floats or float arrays. Got: {k}: {v}{type(v)}"
            assert (
                isinstance(v, float)
                or (isinstance(v, jnp.ndarray) and v.dtype == jnp.float32)
                or (isinstance(v, np.ndarray) and v.dtype == np.float32)
            ), msg
    else:
        for v in params.values():
            validate_params(v)


def copy_params(params: ParamsDict) -> ParamsDict:
    """copy a parameter dictionary

    Args:
        params: the parameter dictionary to copy

    Returns:
        the copied parameter dictionary

    Note:
        this copy function works recursively on all subdictionaries of the params
        dictionary but does NOT copy any non-dictionary values.
    """
    validate_params(params)
    params = {**params}
    if all(isinstance(v, dict) for v in params.values()):
        return {k: copy_params(params[k]) for k in params}
    return params


def set_global_params(params: ParamsDict, **kwargs) -> ParamsDict:
    """add or update the given keyword arguments to each (sub)dictionary of the
       given params dictionary

    Args:
        params: the parameter dictionary to update with the given global parameters
        **kwargs: the global parameters to update the parameter dictionary with.
            These global parameters are often wavelength ('wl') or temperature ('T').

    Returns:
        The modified dictionary.

    Note:
        This operation NEVER updates the given params dictionary inplace.

    Example:
        This is how to change the wavelength to 1600nm for each component in
        the nested parameter dictionary::

            params = set_global_params(params, wl=1.6e-6)
    """
    validate_params(params)
    params = copy_params(params)
    if all(isinstance(v, dict) for v in params.values()):
        return {k: set_global_params(params[k], **kwargs) for k in params}
    params.update(kwargs)
    validate_params(params)
    return params


def get_ports(model: ModelDict) -> Tuple[str, ...]:
    """get port names of the model

    Args:
        model: the model dictionary to get the port names from
    """
    ports: Dict[str, Any] = {}
    for key in model:
        if isinstance(key, str):
            continue
        p1, p2 = key
        ports[p1] = None
        ports[p2] = None
    return tuple(p for p in ports)


def rename_ports(
    model: ModelDict, ports: Union[Dict[str, str], Tuple[str]]
) -> ModelDict:
    """rename the ports of a model

    Args:
        model: the model dictionary to rename the ports for
        ports: a port mapping (dictionary) with keys the old names and values
            the new names.
    """
    original_ports = get_ports(model)
    assert len(ports) == len(original_ports)
    if not isinstance(ports, dict):
        assert len(ports) == len(set(ports))
        ports = {original_ports[i]: port for i, port in enumerate(ports)}
    new_model: ModelDict = {}
    for key in model:
        if isinstance(key, str):
            value = model[key]
            if isinstance(value, dict):
                value = {**value}
            new_model[key] = value
        else:
            p1, p2 = key
            new_model[ports[p1], ports[p2]] = model[p1, p2]
    return new_model


def zero(params: ParamsDict) -> float:
    """the zero model function.

    Args:
        params: the model parameters dictionary

    Returns:
        This function always returns zero.
    """
    return 0.0


def cartesian_product(*arrays) -> jnp.ndarray:
    """calculate the n-dimensional cartesian product, i.e. create all
       possible combinations of all elements in a given collection of arrays.

    Args:
        *arrays:  the arrays to calculate the cartesian product for

    Returns:
        the cartesian product.
    """
    ixarrays = jnp.ix_(*arrays)
    barrays = jnp.broadcast_arrays(*ixarrays)
    sarrays = jnp.stack(barrays, -1)
    product = sarrays.reshape(-1, sarrays.shape[-1])
    return product
