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
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "pycroscope"
copyright = "2021, Jelle Zijlstra"
author = "Jelle Zijlstra"

# The full version, including alpha/beta/rc tags
try:
    release = package_version("pycroscope")
except PackageNotFoundError:
    release = "unknown"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "myst_parser",
]

# If you need extensions of a certain version or higher, list them here.
needs_extensions = {"myst_parser": "0.13.7"}

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

nitpick_ignore = [
    ("py:class", "inspect._empty"),
    ("py:class", "pycroscope.checker.Checker"),
    ("py:class", "pycroscope.error_code.Error"),
    ("py:class", "pycroscope.extensions._T"),
    ("py:class", "pycroscope.find_unused.UnusedObjectFinder"),
    ("py:class", "pycroscope.functions.FunctionInfo"),
    ("py:class", "pycroscope.input_sig.ActualArguments"),
    ("py:class", "pycroscope.input_sig.ParamSpecSig"),
    ("py:class", "pycroscope.node_visitor.ErrorContext"),
    ("py:class", "pycroscope.node_visitor.Failure"),
    ("py:class", "pycroscope.options.Options"),
    ("py:class", "pycroscope.options.T"),
    ("py:class", "pycroscope.signature.CheckCallContext"),
    ("py:class", "pycroscope.stacked_scopes.ConstraintType"),
    ("py:class", "pycroscope.stacked_scopes.ScopeType"),
    ("py:class", "pycroscope.stacked_scopes.T"),
    ("py:class", "pycroscope.type_evaluation.Evaluator"),
    ("py:class", "pycroscope.type_object.TypeObject"),
    ("py:class", "pycroscope.typeshed.TypeshedFinder"),
    ("py:class", "pycroscope.value.OverlapMode"),
    ("py:class", "pycroscope.value.T"),
]


autodoc_member_order = "bysource"
autodoc_default_options = {"inherited-members": False, "member-order": "bysource"}
autodoc_inherit_docstrings = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]
