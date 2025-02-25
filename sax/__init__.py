"""SAX: S + Autograd + XLA."""

from __future__ import annotations

__author__ = "Floris Laporte"
__version__ = "0.14.2"


from functools import partial
from math import pi

from scipy.constants import c

from . import backends, models, saxtypes, utils
from .circuit import circuit, get_required_circuit_models
from .loss import huber_loss, l2_reg, mse
from .models import get_models, passthru
from .multimode import multimode, singlemode
from .netlist import (
    Netlist,
    RecursiveNetlist,
    flatten_netlist,
    get_component_instances,
    get_netlist_instances_by_prefix,
    load_netlist,
    load_recursive_netlist,
    netlist,
    rename_instances,
    rename_models,
)
from .saxtypes import (
    Array,
    ArrayLike,
    Complex,
    ComplexArray1D,
    ComplexArrayND,
    Float,
    FloatArray1D,
    FloatArrayND,
    Int,
    IntArray1D,
    IntArrayND,
    Model,
    ModelFactory,
    PortCombination,
    PortMap,
    SCoo,
    SDense,
    SDict,
    Settings,
    SType,
    is_complex,
    is_complex_float,
    is_float,
    is_mixedmode,
    is_model,
    is_model_factory,
    is_multimode,
    is_scoo,
    is_sdense,
    is_sdict,
    is_singlemode,
    modelfactory,
    scoo,
    sdense,
    sdict,
    validate_model,
)
from .utils import (
    block_diag,
    cartesian_product,
    clean_string,
    copy_settings,
    denormalize,
    flatten_dict,
    get_inputs_outputs,
    get_port_combinations,
    get_ports,
    get_settings,
    grouped_interp,
    merge_dicts,
    mode_combinations,
    normalization,
    normalize,
    reciprocal,
    rename_params,
    rename_ports,
    try_complex_float,
    unflatten_dict,
    update_settings,
    validate_multimode,
    validate_not_mixedmode,
    validate_sdict,
    validate_settings,
)
