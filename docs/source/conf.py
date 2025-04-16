# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))  # Point Sphinx to the project root

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "molE"
copyright = "2025, MSchauperl"
author = "MSchauperl"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.intersphinx",  # Link to other projects' docs (e.g. Python, NumPy)
    "sphinx.ext.viewcode",  # Add links to source code from documentation
]

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pytorch_lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
}
intersphinx_disabled_domains = ["std"]  # Optional: Disable certain domains if needed

templates_path = ["_templates"]
exclude_patterns = []

# autodoc/autosummary settings
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
