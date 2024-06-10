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
import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path.absolute()))

# -- Project information -----------------------------------------------------

project = "aiaccel"
copyright = "2024, AIST"
author = "AIST"

# The full version, including alpha/beta/rc tags
release = "1.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_fontawesome",
    "sphinx_rtd_theme",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# Auto-generated header anchors
# The MyST Parser can automatically generate label “slugs” for header anchors
# so that you can reference them from markdown links. For example, you can use
# header bookmark links, locally; [](#header-anchor), or cross-file
# [](path/to/file.md#header-anchor). To achieve this, use the
# myst_heading_anchors = DEPTH configuration option, where DEPTH is the depth
# of header levels for which you wish to generate links.
# myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False
html_show_sphinx = False
# html_static_path = ["_static"]
html_sidebars = {"**": ["search-field", "sidebar-nav-bs", "sidebar-ethical-ads"]}
source_suffix = {".rst": "restructuredtext", ".txt": "markdown", ".md": "markdown"}
html_context = {
    "support_languages": {
        "ja": "Japanese",
        "en": "English",
    }
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

gettext_compact = False
locale_dirs = ["locale/"]
language = "ja"

# -- Extension configuration -------------------------------------------------
todo_include_todos = True
# autoclass_content = "both"
