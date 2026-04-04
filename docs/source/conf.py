# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

# Configuration file for the Sphinx documentation builder.
# Full options: https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys

import aiaccel

# -- Path setup --------------------------------------------------------------
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path.absolute()))

# -- Project information -----------------------------------------------------
project = "aiaccel"
author = "National Institute of Advanced Industrial Science And Technology (AIST)"
project_copyright = author
release = aiaccel.__version__

html_logo = f"{root_path}/docs/image/logo_aiaccel.png"
html_favicon = f"{root_path}/docs/image/favicon.ico"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_design",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

language = "en"

# -- HTML output -------------------------------------------------------------
html_theme = "shibuya"
html_show_sourcelink = False
html_show_sphinx = False

html_static_path = ["_static"]
html_theme_options = {
    "accent_color": "gold",
    "nav_links": [
        {"title": "User Guide", "url": "user_guide/index"},
        {
            "title": "API Reference",
            "url": "api_reference/index",
            "children": [
                {
                    "title": "OmegaConf Utilities",
                    "url": "api_reference/config",
                    "summary": "aiaccel.config",
                },
                {
                    "title": "PyTorch/Lightning Toolkit",
                    "url": "api_reference/torch",
                    "summary": "aiaccel.torch",
                },
                {
                    "title": "Hyperparameter Optimization",
                    "url": "api_reference/hpo",
                    "summary": "aiaccel.hpo",
                },
            ],
        },
        {"title": "Contribution Guide", "url": "contribution_guide/index"},
    ],
    "github_url": "https://github.com/aistairc/aiaccel",
    "globaltoc_expand_depth": 1,
}
