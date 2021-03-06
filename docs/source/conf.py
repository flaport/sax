# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

from __future__ import annotations

import os
import sys
import shutil

sys.path.insert(
    0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
import sax
import sax._typing


# -- Patch SAX objects -------------------------------------------------------

sax._typing.Model.__repr__ = lambda self: "Model"


# -- Project information -----------------------------------------------------

project = "SAX"
copyright = "2021, Floris Laporte"
author = "Floris Laporte"

# The full version, including alpha/beta/rc tags
release = sax.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Change how type hints are displayed (requires sphinx.ext.autodoc.typehints)
autodoc_typehints = "signature"  # signature, description, none
autodoc_type_aliases = {t: f"sax._typing.{t}" for t in sax._typing.__all__}

# -- Examples Folder ---------------------------------------------------------

sourcedir = os.path.dirname(__file__)
staticdir = os.path.join(sourcedir, "_static")
sax_src = os.path.abspath(os.path.join(sourcedir, "..", "..", "sax"))
examples_src = os.path.abspath(os.path.join(sourcedir, "..", "..", "examples"))
examples_dst = os.path.abspath(os.path.join(sourcedir, "examples"))
os.makedirs(staticdir, exist_ok=True)
shutil.rmtree(examples_dst, ignore_errors=True)
shutil.copytree(examples_src, examples_dst)
shutil.copytree(sax_src, os.path.join(examples_dst, "sax"))
