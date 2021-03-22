""" Utilities for neural network models """

from __future__ import annotations

import os
import json

import jax
import jax.numpy as jnp

from typing import Dict
from .._typing import ComplexFloat


def load_json_weights(path: str) -> Dict[str, ComplexFloat]:
    path = os.path.abspath(os.path.expanduser(path))
    weights = {}
    if os.path.exists(path):
        with open(path, "r") as file:
            for k, v in json.load(file).items():
                _v = jnp.array(v, dtype=float)
                assert isinstance(_v, jnp.ndarray)
                weights[k] = _v
    return weights


def save_json_weights(weights: Dict[str, jnp.ndarray], path: str):
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "w") as file:
        json.dump({k: v.tolist() for k, v in weights.items()}, file)


def generate_random_weights(key, *layer_shapes):
    if isinstance(key, int):
        key = jax.random.PRNGKey(key)
    keys = jax.random.split(key, 2 * len(layer_shapes))
    rand = jax.nn.initializers.lecun_normal()
    weights = {}
    for i, (m, n) in enumerate(zip(layer_shapes[:-1], layer_shapes[1:])):
        weights[f"w{i}"] = rand(keys[2 * i], (m, n))
        weights[f"b{i}"] = rand(keys[2 * i + 1], (1, n))
    return weights


def normalize(x, mean=0.0, std=1.0):
    return (x - mean) / std


def denormalize(x, mean=0.0, std=1.0):
    return x * std + mean


def get_normalization(x):
    return x.mean(0, keepdims=True), x.std(0, keepdims=True)
