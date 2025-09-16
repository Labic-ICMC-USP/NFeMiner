# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# Adds the root directory of your project (where 'nfeminer' is) to sys.path.
# Since 'conf.py' is inside 'docs/source/', we need to go up two levels:
# (docs/source/ -> docs/ -> NFeMiner/)
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NFeMiner'
copyright = '2025, .'
author = '.'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Automatically document Python code
    'sphinx.ext.napoleon',      # Support for Google/NumPy-style docstrings
    'sphinx.ext.viewcode',      # Add links to source code
    'sphinx.ext.intersphinx',   # Link to documentation of other projects
    'sphinx.ext.todo',          # If you use the '.. todo::' directive
]

# Settings for the Napoleon extension (if using Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = []

# Prevent autodoc from trying to import heavy/unstable libraries
autodoc_mock_imports = [
    "unsloth",
    "unsloth_zoo",
    "datasets",
    "transformers",
    "trl",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
