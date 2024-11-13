""" SAX: S + Autograd + XLA """

from __future__ import annotations

__author__ = "Floris Laporte"
__version__ = "0.14.0"


from functools import partial as partial
from math import pi as pi

from scipy.constants import c as c

from . import backends as backends
from . import models as models
from . import saxtypes as saxtypes
from . import utils as utils
from .circuit import circuit as circuit
from .circuit import get_required_circuit_models as get_required_circuit_models
from .loss import huber_loss as huber_loss
from .loss import l2_reg as l2_reg
from .loss import mse as mse
from .models import get_models as get_models
from .models import passthru as passthru
from .multimode import multimode as multimode
from .multimode import singlemode as singlemode
from .netlist import Netlist as Netlist
from .netlist import RecursiveNetlist as RecursiveNetlist
from .netlist import flatten_netlist as flatten_netlist
from .netlist import get_component_instances as get_component_instances
from .netlist import get_netlist_instances_by_prefix as get_netlist_instances_by_prefix
from .netlist import load_netlist as load_netlist
from .netlist import load_recursive_netlist as load_recursive_netlist
from .netlist import netlist as netlist
from .saxtypes import Array as Array
from .saxtypes import ArrayLike as ArrayLike
from .saxtypes import Complex as Complex
from .saxtypes import ComplexArray1D as ComplexArray1D
from .saxtypes import ComplexArrayND as ComplexArrayND
from .saxtypes import Float as Float
from .saxtypes import FloatArray1D as FloatArray1D
from .saxtypes import FloatArrayND as FloatArrayND
from .saxtypes import Int as Int
from .saxtypes import IntArray1D as IntArray1D
from .saxtypes import IntArrayND as IntArrayND
from .saxtypes import Model as Model
from .saxtypes import ModelFactory as ModelFactory
from .saxtypes import PortCombination as PortCombination
from .saxtypes import PortMap as PortMap
from .saxtypes import SCoo as SCoo
from .saxtypes import SDense as SDense
from .saxtypes import SDict as SDict
from .saxtypes import Settings as Settings
from .saxtypes import SType as SType
from .saxtypes import is_complex as is_complex
from .saxtypes import is_complex_float as is_complex_float
from .saxtypes import is_float as is_float
from .saxtypes import is_mixedmode as is_mixedmode
from .saxtypes import is_model as is_model
from .saxtypes import is_model_factory as is_model_factory
from .saxtypes import is_multimode as is_multimode
from .saxtypes import is_scoo as is_scoo
from .saxtypes import is_sdense as is_sdense
from .saxtypes import is_sdict as is_sdict
from .saxtypes import is_singlemode as is_singlemode
from .saxtypes import modelfactory as modelfactory
from .saxtypes import scoo as scoo
from .saxtypes import sdense as sdense
from .saxtypes import sdict as sdict
from .saxtypes import validate_model as validate_model
from .utils import block_diag as block_diag
from .utils import cartesian_product as cartesian_product
from .utils import clean_string as clean_string
from .utils import copy_settings as copy_settings
from .utils import denormalize as denormalize
from .utils import flatten_dict as flatten_dict
from .utils import get_inputs_outputs as get_inputs_outputs
from .utils import get_port_combinations as get_port_combinations
from .utils import get_ports as get_ports
from .utils import get_settings as get_settings
from .utils import grouped_interp as grouped_interp
from .utils import merge_dicts as merge_dicts
from .utils import mode_combinations as mode_combinations
from .utils import normalization as normalization
from .utils import normalize as normalize
from .utils import reciprocal as reciprocal
from .utils import rename_params as rename_params
from .utils import rename_ports as rename_ports
from .utils import try_complex_float as try_complex_float
from .utils import unflatten_dict as unflatten_dict
from .utils import update_settings as update_settings
from .utils import validate_multimode as validate_multimode
from .utils import validate_not_mixedmode as validate_not_mixedmode
from .utils import validate_sdict as validate_sdict
from .utils import validate_settings as validate_settings
