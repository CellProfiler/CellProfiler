# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

import sphinx_rtd_theme


sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src", "subpackages", "library", "cellprofiler_library")))
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src", "subpackages", "library", "cellprofiler_library", "functions")))
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src", "subpackages", "library", "cellprofiler_library", "modules")))
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src", "subpackages", "library", "cellprofiler_library", "opts")))

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


import inspect

def simplify_signature(app, what, name, obj, options, signature, return_annotation):
    import inspect

    if signature is None:
        return None

    try:
        sig = inspect.signature(obj)
    except Exception:
        return signature, ""  # Remove return annotation anyway

    param_names = [param.name for param in sig.parameters.values()]
    simple_sig = "(" + ", ".join(param_names) + ")"

    # Return signature with no return annotation
    return simple_sig, ""
def setup(app):
    app.connect("autodoc-process-signature", simplify_signature)