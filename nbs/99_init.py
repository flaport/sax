# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: sax
#     language: python
#     name: sax
# ---

# +
# default_exp __init__
# -

# # SAX init
#
# > Import everything into the `sax` namespace:

# +
# hide
import matplotlib.pyplot as plt
from fastcore.test import test_eq
from pytest import approx, raises

import os, sys; sys.stderr = open(os.devnull, "w")

# +
# exporti
from __future__ import annotations

__author__ = "Floris Laporte"
__version__ = "0.8.8"
# -

# ## External
#
# utils from other packages available in SAX for convenience:

# +
# exports
from functools import partial as partial
from math import pi as pi

from scipy.constants import c as c

try:
    from flax.core.frozen_dict import FrozenDict as FrozenDict
except ImportError:
    FrozenDict = dict
# -

# ## Typing

# +
# exports

from sax import typing_ as typing
from sax.typing_ import (
    Array as Array,
    ComplexFloat as ComplexFloat,
    Float as Float,
    Model as Model,
    ModelFactory as ModelFactory,
    Models as Models,
    SCoo as SCoo,
    SDense as SDense,
    SDict as SDict,
    Settings as Settings,
    SType as SType,
    is_complex as is_complex,
    is_complex_float as is_complex_float,
    is_float as is_float,
    is_mixedmode as is_mixedmode,
    is_model as is_model,
    is_model_factory as is_model_factory,
    is_multimode as is_multimode,
    is_scoo as is_scoo,
    is_sdense as is_sdense,
    is_sdict as is_sdict,
    is_singlemode as is_singlemode,
    modelfactory as modelfactory,
    scoo as scoo,
    sdense as sdense,
    sdict as sdict,
    validate_model as validate_model,
)
# -

# ## Utils

# +
# exports

from sax import utils as utils
from sax.utils import (
    block_diag as block_diag,
    clean_string as clean_string,
    copy_settings as copy_settings,
    flatten_dict as flatten_dict,
    get_inputs_outputs as get_inputs_outputs,
    get_port_combinations as get_port_combinations,
    get_ports as get_ports,
    get_settings as get_settings,
    grouped_interp as grouped_interp,
    merge_dicts as merge_dicts,
    mode_combinations as mode_combinations,
    reciprocal as reciprocal,
    rename_params as rename_params,
    rename_ports as rename_ports,
    try_float as try_float,
    unflatten_dict as unflatten_dict,
    update_settings as update_settings,
    validate_multimode as validate_multimode,
    validate_not_mixedmode as validate_not_mixedmode,
    validate_sdict as validate_sdict,
    validate_settings as validate_settings,
)
# -

# ## Multimode

# +
# exports

from sax import multimode as multimode
from sax.multimode import (
    multimode as multimode,
    singlemode as singlemode,
)
# -

# ## Models

# +
# exports

from sax import models as models
from sax.models import get_models as get_models, passthru as passthru
# -

# ## Netlist

# +
# exports

from sax.netlist import netlist as netlist
from sax.netlist import load_netlist as load_netlist
from sax.netlist import load_recursive_netlist as load_recursive_netlist
# -

# ## Circuit

# +
# exports

from sax.circuit import circuit as circuit
from sax.circuit import get_required_circuit_models as get_required_circuit_models
# -

# ## Backend

# +
# exports

from sax import backends as backends
# -

# ## Patches

# +
# exports

from sax import patched as _patched
