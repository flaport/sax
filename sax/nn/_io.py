""" IO utilities for SAX neural networks """

from __future__ import annotations

import os
import re
import json

import jax.numpy as jnp


from .nn import dense, preprocess
from .utils import norm

from typing import Tuple, List, Optional, Dict, Callable
from .._typing import ComplexFloat


def load_nn_weights_json(path: str) -> Dict[str, ComplexFloat]:
    """Load json weights from given path

    Args:
        path: the path to load the json weights from

    Returns:
        a dictionary of weights
    """
    path = os.path.abspath(os.path.expanduser(path))
    weights = {}
    if os.path.exists(path):
        with open(path, "r") as file:
            for k, v in json.load(file).items():
                _v = jnp.array(v, dtype=float)
                assert isinstance(_v, jnp.ndarray)
                weights[k] = _v
    return weights


def save_nn_weigths_json(weights: Dict[str, ComplexFloat], path: str):
    """Save json weights to given path

    Args:
        weights: dictionary of weights to save as json
        path: the path to load the json weights from

    """
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        _weights = {}
        for k, v in weights.items():
            v = jnp.atleast_1d(jnp.array(v))
            assert isinstance(v, jnp.ndarray)
            _weights[k] = v.tolist()
        json.dump(_weights, file)


def get_available_sizes(
    dirpath: str,
    prefix: str,
    input_names: Tuple[str, ...],
    output_names: Tuple[str, ...],
) -> List[Tuple[int, ...]]:
    """Get all available hidden sizes given filename parameters

    Args:
        prefix: the prefix of the filenames to check
        dirpath: the folder within to check for matching files
        input_names: the input feature names
        outptut_names: the output (predicted) feature names

    Returns:
        the possible hidden sizes

    Note:
        this function does NOT return the input size and the output size of the
        neural network. ONLY the hidden sizes are reported. The input and
        output sizes can easily be derived from `input_names` (after
        preprocessing) and `output_names`.
    """
    all_weightfiles = os.listdir(dirpath)
    possible_weightfiles = (
        s for s in all_weightfiles if s.endswith(f"-{'-'.join(output_names)}.json")
    )
    possible_weightfiles = (
        s
        for s in possible_weightfiles
        if s.startswith(f"{prefix}-{'-'.join(input_names)}")
    )
    possible_weightfiles = (re.sub("[^0-9x]", "", s) for s in possible_weightfiles)
    possible_weightfiles = (re.sub("^x*", "", s) for s in possible_weightfiles)
    possible_weightfiles = (re.sub("x[^0-9]*$", "", s) for s in possible_weightfiles)
    possible_hidden_sizes = (s.strip() for s in possible_weightfiles if s.strip())
    possible_hidden_sizes = (
        tuple(hs.strip() for hs in s.split("x") if hs.strip())
        for s in possible_hidden_sizes
    )
    possible_hidden_sizes = (
        tuple(int(hs) for hs in s[1:-1]) for s in possible_hidden_sizes if len(s) > 2
    )
    possible_hidden_sizes = sorted(
        possible_hidden_sizes, key=lambda hs: (len(hs), max(hs))
    )
    return possible_hidden_sizes


def get_dense_weights_path(
    *sizes: int,
    input_names: Optional[Tuple[str, ...]] = None,
    output_names: Optional[Tuple[str, ...]] = None,
    dirpath: str = "weights",
    prefix: str = "dense",
    preprocess=preprocess,
):
    """Create the SAX conventional path for a given dictionary

    Args:
        *sizes: the sizes of the dense neural network weights. This can be left
            out for a single layer neural network when specifying both
            `input_names` and `output_names`.
        input_names: the names of the input parameters of the neural network.
            If given, the input dimension of the neural network will be derived
            from the number of features after preprocessing these names, i.e. this
            input dimension will then be prepended to `sizes`.
        output_names: the names of the output parameters the neural network
            should predict. If given, the output dimension of the neural
            network will be set to the number of output names, i.e. th number
            of output names will be appended to `sizes`.
        dirpath: the folder to save the weights to
        prefix: the prefix to give the weights filename
        preprocess: the preprocessing function which will be used in the neural
            network. This preprocessing function determines the input dimension
            (if `input_names` is given.)
    """
    if input_names:
        num_inputs = preprocess(*jnp.ones(len(input_names))).shape[0]
        sizes = (num_inputs,) + sizes
    if output_names:
        sizes = sizes + (len(output_names),)
    path = os.path.abspath(os.path.join(dirpath, prefix))
    if input_names:
        path = f"{path}-{'-'.join(input_names)}"
    if sizes:
        path = f"{path}-{'x'.join(str(s) for s in sizes)}"
    if output_names:
        path = f"{path}-{'-'.join(output_names)}"
    return f"{path}.json"


def get_norm_path(
    *shape: int,
    input_names: Optional[Tuple[str, ...]] = None,
    output_names: Optional[Tuple[str, ...]] = None,
    dirpath: str = "norms",
    prefix: str = "norm",
    preprocess=preprocess,
):
    """Create the SAX conventional path for the normalization constants

    Args:
        *shape: the shape of the norm to create the path for. This can be left
            out if `input_names` or `output_names` is specified.
        input_names: the names of the input parameters of the neural network.
            If given, the shape of the norm will be prepended with the number
            of features after preprocessing these names. (mutually exclusive
            with `output_names`)
        output_names: the names of the output parameters the neural network
            should predict. If given, the shape of the norm will be appended
            with the number of output names. (mutually exclusive with
            `input_names`)
        dirpath: the folder to save the norms to
        prefix: the prefix to give the norm filename
        preprocess: the preprocessing function which will be used in the neural
            network. This preprocessing function determines the input dimension
            (if `input_names` is given.)
    """
    if input_names and output_names:
        raise ValueError(
            "To get the norm name, one can only specify `input_names` OR `output_names`."
        )
    if input_names:
        num_inputs = preprocess(*jnp.ones(len(input_names))).shape[0]
        shape = (num_inputs,) + shape
    if output_names:
        shape = shape + (len(output_names),)
    path = os.path.abspath(os.path.join(dirpath, prefix))
    if input_names:
        path = f"{path}-{'-'.join(input_names)}"
    if shape:
        path = f"{path}-{'x'.join(str(s) for s in shape)}"
    if output_names:
        path = f"{path}-{'-'.join(output_names)}"
    return f"{path}.json"


def load_nn_dense(
    *sizes: int,
    input_names: Optional[Tuple[str, ...]] = None,
    output_names: Optional[Tuple[str, ...]] = None,
    weightprefix="dense",
    weightdirpath="weights",
    normdirpath="norms",
    normprefix="norm",
    preprocess=preprocess,
) -> Callable:
    """Load a pre-trained dense model

    Args:
        *sizes: the sizes of the dense neural network weights. This can be left
            out for a single layer neural network when specifying both
            `input_names` and `output_names`.
        input_names: the names of the input parameters of the neural network.
            If given, the input dimension of the neural network will be derived
            from the number of features after preprocessing these names, i.e. this
            input dimension will then be prepended to `sizes`.
        output_names: the names of the output parameters the neural network
            should predict. If given, the output dimension of the neural
            network will be set to the number of output names, i.e. th number
            of output names will be appended to `sizes`.
        weightdirpath: the folder to save the weights to
        weightprefix: the prefix to give the weights filename
        normdirpath: the folder to save the norms to
        normprefix: the prefix to give the norm filename
        preprocess: the preprocessing function which will be used in the neural
            network. This preprocessing function determines the input dimension
            (if `input_names` is given.)
    """
    weights_path = get_dense_weights_path(
        *sizes,
        input_names=input_names,
        output_names=output_names,
        prefix=weightprefix,
        dirpath=weightdirpath,
        preprocess=preprocess,
    )
    if not os.path.exists(weights_path):
        raise ValueError("Cannot find weights path for given parameters")
    x_norm_path = get_norm_path(
        input_names=input_names,
        prefix=normprefix,
        dirpath=normdirpath,
        preprocess=preprocess,
    )
    if not os.path.exists(x_norm_path):
        raise ValueError("Cannot find normalization for input parameters")
    y_norm_path = get_norm_path(
        output_names=output_names,
        prefix=normprefix,
        dirpath=normdirpath,
        preprocess=preprocess,
    )
    if not os.path.exists(x_norm_path):
        raise ValueError("Cannot find normalization for output parameters")
    weights = load_nn_weights_json(weights_path)
    x_norm_dict = load_nn_weights_json(x_norm_path)
    y_norm_dict = load_nn_weights_json(y_norm_path)
    x_norm = norm(x_norm_dict["mean"], x_norm_dict["std"])
    y_norm = norm(y_norm_dict["mean"], y_norm_dict["std"])
    partial_dense = _PartialDense(weights, x_norm, y_norm, input_names, output_names)
    return partial_dense


class _PartialDense:
    def __init__(self, weights, x_norm, y_norm, input_names, output_names):
        self.weights = weights
        self.x_norm = x_norm
        self.y_norm = y_norm
        self.input_names = input_names
        self.output_names = output_names

    def __call__(self, *params: ComplexFloat) -> ComplexFloat:
        return dense(self.weights, *params, x_norm=self.x_norm, y_norm=self.y_norm)

    def __repr__(self):
        return f"{self.__class__.__name__}{repr(self.input_names)}->{repr(self.output_names)}"
