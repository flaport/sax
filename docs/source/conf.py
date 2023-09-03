from __future__ import annotations

import os
import shutil
import typing
import sys

# Dynamic Config
REPO_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_PATH)
import sax  # noqa
from sax.typing_ import *  # noqa

EXSRC = os.path.join(REPO_PATH, "examples")
EXDST = os.path.join(REPO_PATH, "docs", "source", "examples")
os.makedirs(EXDST, exist_ok=True)

src = {fn: os.path.join(EXSRC, fn) for fn in os.listdir(EXSRC)}
dst = {fn: os.path.join(EXDST, fn) for fn in os.listdir(EXDST)}
for k in list(dst.keys()):
    if k not in src:
        del dst[k]
for k in src:
    if k not in dst and os.path.isfile(src[k]):
        shutil.copy2(os.path.join(EXSRC, k), os.path.join(EXDST, k))


# Static Config
project = "sax"
copyright = "2023, Apache2"
author = "Floris Laporte"
release = "0.10.0"
extensions = [
    "myst_nb",
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.autodoc_pydantic",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "build",
    "extra",
]
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/img/logo.png"
html_theme_options = {
    "logo_only": True,
    "path_to_docs": "docs",
    "repository_branch": "main",
    "repository_url": "https://github.com/flaport/sax",
    "use_download_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "display_version": True,
    "launch_buttons": {
        "notebook_interface": "jupyterlab",
        "binderhub_url": "https://mybinder.org/v2/gh/flaport/sax/HEAD",
        "colab_url": "https://colab.research.google.com",
    },
}
html_css_files = [
    "_static/css/custom.css",
]
autodoc_member_order = "bysource"
napoleon_use_param = True
nbsphinx_timeout = 300
language = "en"
myst_html_meta = {
    "description lang=en": "metadata description",
    "keywords": "Sphinx, MyST",
    "property=og:locale": "en_US",
}
autodoc_pydantic_model_signature_prefix = "class"
autodoc_pydantic_field_signature_prefix = "attribute"
autodoc_pydantic_model_show_config_member = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_typehints = "description"
autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "inherited-members": False,
    "show-inheritance": False,
}
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")
add_module_names = False
nbsphinx_execute = "never"
nb_execution_mode = "off"

autodoc_type_aliases = {
    k: k
    for k, v in vars(sax.typing_).items()
    if isinstance(v, typing._BaseGenericAlias)  # type: ignore
}
