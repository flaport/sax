from recommonmark.transform import AutoStructify

project = "sax"
version = "0.3.1"
copyright = "2020, Floris Laporte"
author = "Floris Laporte"

master_doc = "index"
html_theme = "sphinx_rtd_theme"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

html_static_path = ["_static"]
htmlhelp_basename = project

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_markdown_tables",
    "sphinx.ext.doctest",
    "recommonmark",
]

autodoc_member_order = "bysource"


def setup(app):
    """Set configuration."""
    app.add_config_value(
        "recommonmark_config",
        {"auto_toc_tree_section": "Contents", "enable_eval_rst": True},
        True,
    )
    app.add_transform(AutoStructify)
