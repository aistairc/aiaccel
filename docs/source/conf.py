# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

# Configuration file for the Sphinx documentation builder.
# Full options: https://www.sphinx-doc.org/en/master/usage/configuration.html

from typing import Any

import inspect
from pathlib import Path
import sys
from types import ModuleType

# -- Path setup --------------------------------------------------------------
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path.resolve()))


def _get_release_version() -> str:
    # Import aiaccel only after updating sys.path so inspect.getsourcefile
    # resolves to local repository files instead of site-packages.
    import aiaccel

    return aiaccel.__version__


# -- Project information -----------------------------------------------------
project = "aiaccel"
author = "National Institute of Advanced Industrial Science And Technology (AIST)"
project_copyright = author
release = _get_release_version()

html_logo = f"{root_path}/docs/image/logo_aiaccel.png"
html_favicon = f"{root_path}/docs/image/favicon.ico"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
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

html_context = {
    "source_user": "aistairc",
    "source_repo": "aiaccel",
    "source_version": "main",
}

SOURCE_REPO_URL = f"https://github.com/{html_context['source_user']}/{html_context['source_repo']}"
SOURCE_VERSION = html_context["source_version"]


def _resolve_object(module: ModuleType, fullname: str | None) -> Any | None:
    if not fullname:
        return module
    # Remove source link of __init__
    if fullname.split(".")[-1] == "__init__":
        return None

    obj: Any = module
    for name in fullname.split("."):
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def _get_source_file(obj: Any) -> str | None:
    try:
        return inspect.getsourcefile(obj) or inspect.getfile(obj)
    except Exception:
        return None


def _get_relative_source_path(source_file: str) -> Path | None:
    try:
        return Path(source_file).resolve().relative_to(root_path.resolve())
    except ValueError:
        return None


def _get_linespec(obj: Any) -> str:
    try:
        source_lines, start_line = inspect.getsourcelines(obj)
    except Exception:
        return ""
    end_line = start_line + len(source_lines) - 1
    return f"#L{start_line}-L{end_line}"


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    if domain != "py":
        return None

    module_name = info.get("module")
    if not module_name:
        return None

    # Get object of module_name
    module = sys.modules.get(module_name)
    if module is None:
        return None

    fullname = info.get("fullname")
    obj = _resolve_object(module, fullname)
    if obj is None:
        return None

    source_file = _get_source_file(obj)
    if source_file is None:
        return None

    relative_path = _get_relative_source_path(source_file)
    if relative_path is None:
        return None

    linespec = _get_linespec(obj)
    return f"{SOURCE_REPO_URL}/blob/{SOURCE_VERSION}/{relative_path.as_posix()}{linespec}"
