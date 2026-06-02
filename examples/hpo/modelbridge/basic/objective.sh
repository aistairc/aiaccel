#!/bin/bash
#$-l rt_C.small=1
#$-cwd

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Optional ABCI environment setup.
if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
    if [ -n "${MODELBRIDGE_MODULE_GCC:-}" ]; then
        module load "${MODELBRIDGE_MODULE_GCC}"
    fi
    if [ -n "${MODELBRIDGE_MODULE_PYTHON:-}" ]; then
        module load "${MODELBRIDGE_MODULE_PYTHON}"
    fi
fi

PYTHON_BIN="${MODELBRIDGE_PYTHON:-python}"
OBJECTIVE_SCRIPT="${MODELBRIDGE_OBJECTIVE_SCRIPT:-${SCRIPT_DIR}/objectives/simple_objective.py}"

if [[ "${OBJECTIVE_SCRIPT}" != /* ]]; then
    if [ -f "${SCRIPT_DIR}/${OBJECTIVE_SCRIPT}" ]; then
        OBJECTIVE_SCRIPT="${SCRIPT_DIR}/${OBJECTIVE_SCRIPT}"
    else
        OBJECTIVE_SCRIPT="${PROJECT_ROOT}/${OBJECTIVE_SCRIPT}"
    fi
fi

PIXI_COMMAND="${MODELBRIDGE_PIXI_COMMAND:-pixi}"
PIXI_PROJECT_ROOT="${MODELBRIDGE_PIXI_PROJECT_ROOT:-}"
PIXI_ENVIRONMENT="${MODELBRIDGE_PIXI_ENVIRONMENT:-}"
USE_PIXI="${MODELBRIDGE_USE_PIXI:-}"

if [ -z "${USE_PIXI}" ]; then
    if [ -n "${PIXI_PROJECT_ROOT}" ] || [ -n "${PIXI_ENVIRONMENT}" ]; then
        USE_PIXI=1
    else
        USE_PIXI=0
    fi
fi

if [ "${USE_PIXI}" = "1" ] && command -v "${PIXI_COMMAND}" >/dev/null 2>&1; then
    PIXI_PROJECT_ROOT="${PIXI_PROJECT_ROOT:-${PROJECT_ROOT}}"
    pixi_args=(run --manifest-path "${PIXI_PROJECT_ROOT}/pyproject.toml")
    if [ -n "${PIXI_ENVIRONMENT}" ]; then
        pixi_args+=(-e "${PIXI_ENVIRONMENT}")
    fi
    exec "${PIXI_COMMAND}" "${pixi_args[@]}" "${PYTHON_BIN}" "${OBJECTIVE_SCRIPT}" "$@"
fi

exec "${PYTHON_BIN}" "${OBJECTIVE_SCRIPT}" "$@"
