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
# default_exp nn.__init__
# -

# # Neural Networks
#
# > Utilitites for creating advanced neural network SAX Models

# export
from __future__ import annotations

# hide
import os, sys; sys.stderr = open(os.devnull, "w")

# ## Check

try:
    import jax
except ImportError:
    raise ImportError("[NO JAX] Cannot import sax.nn. Please install JAX first!")

# ## Loss

# +
# exports

from sax.nn.loss import huber_loss as huber_loss
from sax.nn.loss import l2_reg as l2_reg
from sax.nn.loss import mse as mse
# -

# ## Utils

# +
# exports

from sax.nn.utils import (
    cartesian_product as cartesian_product,
    denormalize as denormalize,
    get_normalization as get_normalization,
    get_df_columns as get_df_columns,
    normalize as normalize,
)
# -

# ## Core

# +
# exports

from sax.nn.core import (
    preprocess as preprocess,
    dense as dense,
    generate_dense_weights as generate_dense_weights,
)
# -

# ## IO

# +
# exports

from sax.nn.io import (
    load_nn_weights_json as load_nn_weights_json,
    save_nn_weights_json as save_nn_weights_json,
    get_available_sizes as get_available_sizes,
    get_dense_weights_path as get_dense_weights_path,
    get_norm_path as get_norm_path,
    load_nn_dense as load_nn_dense,
)
