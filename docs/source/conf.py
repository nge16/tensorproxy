# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../pyhydrosym/"))

project = "pyhydrosym"
copyright = "2024, Alexander Belinsky"
author = "Alexander Belinsky"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

source_suffix = ".rst"
master_doc = "index"
autoclass_content = "both"

# Intersphinx mappings.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "ipyparallel": ("https://ipyparallel.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pagmo": ("https://esa.github.io/pagmo2/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

language = "ru"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme = "sphinx_book_theme"
# html_static_path = ["_static"]

# html_favicon = 'images/logo_favico.ico'
# html_logo = "_static/img/text_logo.jpg"

html_theme_options = {
    "repository_url": "https://github.com/abelinsky/proxy-modeling-toolkit",
    "repository_branch": "master",
    "path_to_docs": "doc",
    "use_repository_button": True,
    "use_issues_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
    },
}


nb_execution_mode = "force"

latex_engine = "xelatex"

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]


add_module_names = False


def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = [
            "__init__",
            "__repr__",
            "__weakref__",
            "__dict__",
            "__module__",
        ]
        return True if name in members else skip

    app.connect("autodoc-skip-member", skip)
