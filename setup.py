from setuptools import setup

hpo = [
    "numpy",
    "optuna>=3.4.0",
]
torch = ["hydra-core", "lightning>=2.2.1", "torch>=2.2.0", "h5py"]
dev = [
    "mypy",
    "myst-parser",
    "pre-commit",
    "pytest",
    "pytest-cov",
    "pytest-mock",
    "pytest-subprocess",
    "ruff",
    "sphinx",
    "sphinxcontrib-jquery",
    "sphinx-intl",
    "sphinx-fontawesome",
    "sphinx-rtd-theme",
    "pydata-sphinx-theme",
    "sphinx_design",
    "types-colorama",
    "undecorated",
    "pandas",
    "pandas-stubs",
    "matplotlib",
]
github_actions = [
    "pytest-github-actions-annotate-failures",
]
all_dependencies = hpo + torch
all_dev_dependencies = hpo + torch + dev + github_actions

setup(
    extras_require={
        "all": all_dependencies,
        "all-dev": all_dev_dependencies,
        "hpo": hpo,
        "torch": torch,
        "dev": dev,
        "github-actions": github_actions,
    },
)
