""" Useful functions for working with SAX. """

from __future__ import annotations

import pickle

from typing import (
    Any,
    Union,
    Tuple,
    Dict,
)
from ._typing import (
    ComplexFloat,
    ModelParams,
    ModelDict,
    Model,
    is_complex_float,
)

__all__ = [
    "copy_params",
    "get_ports",
    "rename_ports",
    "set_params",
    "validate_params",
    "zero",
]


def copy_params(params: ModelParams) -> ModelParams:
    """copy a parameter dictionary

    Args:
        params: the parameter dictionary to copy

    Returns:
        the copied parameter dictionary

    Note:
        this copy function works recursively on all subdictionaries of the params
        dictionary but does NOT copy any non-dictionary values.
    """
    _params = {}
    for k, v in params.items():
        if isinstance(v, dict):
            _params[k] = copy_params(v)
        elif is_complex_float(v):
            _params[k] = v
        else:
            raise ValueError(
                "params dictionary to copy does not have the right type format"
            )
    return _params


def get_ports(model: Model) -> Tuple[str, ...]:
    """get port names of the model

    Args:
        model: the model dictionary to get the port names from
    """
    ports: Dict[str, Any] = {}
    for key in model.funcs:
        p1, p2 = key
        ports[p1] = None
        ports[p2] = None
    return tuple(p for p in ports)


def rename_ports(model: Model, ports: Union[Dict[str, str], Tuple[str]]) -> Model:
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
    funcs: ModelDict = {
        (ports[p1], ports[p2]): model.funcs[p1, p2] for p1, p2 in model.funcs
    }
    new_model = Model(funcs=funcs, params=model.params)
    return new_model


def set_params(
    params: ModelParams, *compnames: str, **kwargs: ComplexFloat
) -> ModelParams:
    """update a parameter dictionary

    add or update the given keyword arguments to each (sub)dictionary of the
    given params dictionary

    Args:
        params: the parameter dictionary to update with the given parameters
        *compnames: the nested component names for which to set the parameters.
            If left out, the given parameters will be applied globally to all
            (sub)components in the dictionary.
        **kwargs: the parameters to update the parameter dictionary with.

    Returns:
        The modified dictionary.

    Note:
        - Even though it's possible to update parameter dictionaries in place,
          this function is convenient to apply certain parameters (e.g.
          wavelength 'wl' or temperature 'T') globally.
        - This operation never updates the given params dictionary inplace.
        - Any non-float keyword arguments will be silently ignored.

    Example:
        This is how to change the wavelength to 1600nm for each component in
        the nested parameter dictionary::

            params = set_params(params, wl=1.6e-6)

        Or to set the temperature for only the direcional coupler named 'dc'
        belonging to the MZI named 'mzi' in the circuit::

            params = set_params(params, "mzi", "dc", T=30.0)
    """
    _params = {}
    if not compnames:
        for k, v in params.items():
            if isinstance(v, dict):
                _params[k] = set_params(v, **kwargs)
            elif is_complex_float(v):
                if k in kwargs and is_complex_float(kwargs[k]):
                    _params[k] = kwargs[k]
                else:
                    _params[k] = v
            else:
                raise ValueError(
                    "params dictionary to set global parameters for "
                    "does not have the right type format"
                )
    else:
        for k, v in params.items():
            if isinstance(v, dict):
                if k == compnames[0]:
                    _params[k] = set_params(v, *compnames[1:], **kwargs)
                else:
                    _params[k] = set_params(v)
            elif is_complex_float(v):
                _params[k] = v
            else:
                raise ValueError(
                    "params dictionary to set global parameters for "
                    "does not have the right type format"
                )
    return _params


def validate_params(params: ModelParams):
    """validate a parameter dictionary

    Args:
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
            assert is_complex_float(v), msg
    else:
        for v in params.values():
            assert isinstance(v, dict)
            validate_params(v)


def zero(params: ModelParams) -> float:
    """the zero model function.

    Args:
        params: the model parameters dictionary

    Returns:
        This function always returns zero.
    """
    return 0.0
