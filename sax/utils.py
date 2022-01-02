""" Useful functions for working with SAX. """

from __future__ import annotations

import inspect
from functools import wraps, lru_cache

import jax
import jax.numpy as jnp

from typing import (
    Any,
    Dict,
    Tuple,
    Union,
    overload,
)
from sax._typing import (
    ComplexFloat,
    Model,
    PDict,
    SDict,
    is_complex_float,
)


def copy_params(params: PDict) -> PDict:
    """copy a parameter dictionary (alias of validate_pdict)

    Args:
        params: the parameter dictionary to copy

    Returns:
        the copied parameter dictionary

    Note:
        this copy function works recursively on all subdictionaries of the params
        dictionary but does NOT copy any non-dictionary values. Moreover, it
        also validates the PDict.
    """
    return validate_pdict(params)


def get_params(model: Model) -> PDict:
    """Get the parameters of a SAX model

    Args:
        model: the SAX model (function) to get the parameters for

    Returns:
        the parameter dictionary
    """
    signature = inspect.signature(model)

    params: PDict = {
        k: (v.default if not isinstance(v, dict) else v)
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    # make sure an inplace operation of resulting dict does not change the
    # circuit parameters themselves
    return copy_params(params)


def get_ports(sdict_or_model: Union[Model, SDict]) -> Tuple[str, ...]:
    """get port names of a model or an sdict

    Args:
        sdict: the Model or SDict to get the port names from

    Returns:
        the ports of the model/sdict

    Note:
        if a callable is given, the function will be traced (in much the same
        way as jax.jit) to obtain the keys of the resulting sdict. Although
        this function is 'cheap' in comparison to evaluating the model/circuit.
        It is not for free!  Use this function sparingly on your large circuits!
    """
    if callable(sdict_or_model):
        return _get_ports_from_model(sdict_or_model)
    ports: Dict[str, Any] = {}
    for key in sdict_or_model:
        p1, p2 = key
        ports[p1] = None
        ports[p2] = None
    return tuple(p for p in ports)


def merge_dicts(*dicts: Dict) -> Dict:
    """merge nested dictionaries

    Args:
        *dicts: the dictionaries to merge from left to right, i.e. values in
        dictionaries more to the right get precedence.

    Yields:
        the merged dictionary

    """
    if len(dicts) == 1:
        return dict(_generate_merged_dict(dicts[0], {}))
    elif len(dicts) == 2:
        return dict(_generate_merged_dict(dicts[0], dicts[1]))
    else:
        return merge_dicts(dicts[0], merge_dicts(*dicts[1:]))


def reciprocal(sdict: SDict) -> SDict:
    """Make an SDict reciprocal

    Args:
        sdict: theSDict to make reciprocal

    Returns:
        the reciprocal SDict
    """
    return {
        **{(p1, p2): v for (p1, p2), v in sdict.items()},
        **{(p2, p1): v for (p1, p2), v in sdict.items()},
    }


def rename_params(model: Model, renamings: Dict[str, str]) -> Model:
    """rename the parameters in a model

    Args:
        model: the sax model to rename the parameters for
        renamings: the dictionary mapping of (possibly a selection of) the old
            names to the new names.

    Returns:
        the new sax models with differently named parameters

    """
    validate_model(model)
    old_params = get_params(model)
    old_model = model

    reversed_renamings = {v: k for k, v in renamings.items()}
    if len(reversed_renamings) < len(renamings):
        raise ValueError("Multiple old names point to the same new name!")

    @wraps(old_model)
    def new_model(**params):
        old_params = {reversed_renamings.get(k, k): v for k, v in params.items()}
        return old_model(**old_params)

    new_params = {renamings.get(k, k): v for k, v in old_params.items()}
    _replace_kwargs(new_model, **new_params)

    return new_model


@overload
def rename_ports(sdict_or_model: SDict, ports: Dict[str, str]) -> SDict:
    ...


@overload
def rename_ports(sdict_or_model: Model, ports: Dict[str, str]) -> Model:
    ...


def rename_ports(
    sdict_or_model: Union[SDict, Model], ports: Dict[str, str]
) -> Union[SDict, Model]:
    """rename the ports of an SDict

    Args:
        sdict_or_model: the SDict or Model to rename the ports for
        ports: a port mapping (dictionary) with keys the old names and values
            the new names.
    """
    if isinstance(sdict_or_model, dict):
        sdict = sdict_or_model
        original_ports = get_ports(sdict)
        assert len(ports) == len(original_ports)
        return {(ports[p1], ports[p2]): v for (p1, p2), v in sdict.items()}
    else:
        old_model: Model = sdict_or_model

        @wraps(old_model)
        def new_model(**params):
            return rename_ports(old_model(**params), ports)

        return new_model


def set_params(params: PDict, *compnames: str, **kwargs: ComplexFloat) -> PDict:
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

            params = set_params(params, wl=1.6)

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
                    f"params dict to copy does not have the right type format for '{k}'. "
                    f"Expected dict or float. Got: {type(v)} [{v}]"
                )
    else:
        for k, v in params.items():
            if isinstance(v, dict):
                if k == compnames[0]:
                    _params[k] = set_params(v, *compnames[1:], **kwargs)
                else:
                    _params[k] = v
            elif is_complex_float(v):
                _params[k] = v
            else:
                raise ValueError(
                    f"params dict to copy does not have the right type format for '{k}'. "
                    f"Expected dict or float. Got: {type(v)} [{v}]"
                )
    return _params


def validate_model(model):
    """Validate the parameters of a model

    Args:
        model: the model function to validate

    Note:
        This function checks if all parameters (kwargs) of a model function are
        keyword arguments. If any of the arguments is a positional argument a
        ValueError is raised.
    """
    positional_arguments = []
    for param in inspect.signature(model).parameters.values():
        if param.default is inspect.Parameter.empty:
            positional_arguments.append(param.name)
    if positional_arguments:
        raise ValueError(
            f"model '{model}' takes positional arguments {', '.join(positional_arguments)} "
            "and hence is not a valid SAX Model! "
            "A SAX model should ONLY take keyword arguments (or no arguments at all)."
        )


def validate_pdict(params: PDict) -> PDict:
    """Validate a parameter dictionary

    Args:
        params: the parameter dictionary to validate

    Note:
        this operation also copies the PDict
    """
    _params = {}
    for k, v in params.items():
        try:
            v = jnp.asarray(v, dtype=float)
        except (ValueError, TypeError):
            pass
        if is_complex_float(v):
            _params[k] = v
        elif isinstance(v, dict):
            _params[k] = validate_pdict(v)
        else:
            raise ValueError(
                f"Invalid PDict! A PDict should be a (possibly nested) dictionary of "
                f"floats or float arrays. Got a param {k}={v} (of type {type(v)})."
            )
    return _params


def validate_sdict(sdict: SDict):
    """Validate an SDict

    Args:
        sdict: the sdict to validate
    """
    for k, v in sdict.items():
        if not isinstance(k, tuple) and not len(k) == 2:
            raise ValueError(f"SDict keys should be length-2 tuples. Got {k}")
        if not is_complex_float(v):
            raise ValueError(
                f"SDict values should be complex or floats (or complex array / float "
                f"arrays). Got {k}: {v} [type: {type(v)}]."
            )


def _generate_merged_dict(dict1: Dict, dict2: Dict) -> Dict:
    """merge two (possibly deeply nested) dictionaries

    Args:
        dict1: the first dictionary to merge
        dict2: the second dictionary to merge

    Yields:
        the merged dictionary

    """
    # from https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(_generate_merged_dict(dict1[k], dict2[k])))
            else:
                # If one of the values is not a dict, you can't continue merging it.
                # Value from second dict overrides one in first and we move on.
                yield (k, dict2[k])
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])


@lru_cache(maxsize=4096)  # cache to prevent future tracing
def _get_ports_from_model(model: Model) -> Tuple[str, ...]:
    """get port names of an sdict

    Args:
        model: the Model to get the port names from

    Returns:
        the ports of the model

    Note:
        The given model function will be traced (in much the same
        way as jax.jit) to obtain the keys of the resulting sdict. Although
        this function is 'cheap' in comparison to evaluating the model/circuit.
        It is not for free!  Use this function sparingly on your large circuits!
    """
    traced_sdict = jax.eval_shape(model)
    return get_ports(traced_sdict)


def _replace_kwargs(func, **kwargs):
    """Change the kwargs signature of a function

    Args:
        func: the keyword-only function that takes **kwargs as argument
        **kwargs: the names and values to replace the **kwargs of the function with

    Note:
        this is an auxiliary function for sax.circuit. Use at your own risk.

    """
    sig = inspect.signature(func)
    params = [
        inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=v)
        for k, v in kwargs.items()
    ]
    func.__signature__ = sig.replace(parameters=params, return_annotation=SDict)
    return func
