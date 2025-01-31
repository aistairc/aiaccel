from setuptools import setup

hpo_dependencies = [
    "numpy",
    "optuna>=3.4.0",
]
torch_dependencies = ["hydra-core", "lightning>=2.2.1", "torch>=2.2.0", "h5py", "tensorboard"]
dev_dependencies = [
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
    "docstrfmt",
]
github_actions_dependencies = [
    "pytest-github-actions-annotate-failures",
]
all_dependencies = hpo_dependencies + torch_dependencies
all_dev_dependencies = hpo_dependencies + torch_dependencies + dev_dependencies + github_actions_dependencies

setup(
    extras_require={
        "all": all_dependencies,
        "all-dev": all_dev_dependencies,
        "hpo": hpo_dependencies,
        "torch": torch_dependencies,
        "dev": dev_dependencies,
        "github-actions": github_actions_dependencies,
    },
)
