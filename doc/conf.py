# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "dkpy"
copyright = "2024, Steven Dahdah and James Richard Forbes"
author = "Steven Dahdah and James Richard Forbes"
version = "0.1.4"
release = "0.1.4"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_rtd_theme",
]

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "groupwise",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_typehints = "description"

autosummary_generate = True

# Path to ``objects.inv``
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "control": ("https://python-control.readthedocs.io/en/latest/", None),
    "cvxpy": ("https://www.cvxpy.org/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
}
html_static_path = ["_static"]
