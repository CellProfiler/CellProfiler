# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "..", "subpackages", "library", "cellprofiler_library")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CellProfiler-library'
copyright = '2025, Suraj Pathak'
author = 'Suraj Pathak'
release = '5.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    'sphinx-pydantic',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
autodoc_typehints = "description"