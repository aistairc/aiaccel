#!/bin/bash
#$-l rt_C.small=1
#$-cwd

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

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

if [ -n "${MODELBRIDGE_VENV:-}" ] && [ -f "${MODELBRIDGE_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${MODELBRIDGE_VENV}/bin/activate"
fi

PYTHON_BIN="${MODELBRIDGE_PYTHON:-python3}"
OBJECTIVE_SCRIPT="${MODELBRIDGE_OBJECTIVE_SCRIPT:-${SCRIPT_DIR}/objectives/simple_objective.py}"

if [[ "${OBJECTIVE_SCRIPT}" != /* ]]; then
    OBJECTIVE_SCRIPT="${PROJECT_ROOT}/${OBJECTIVE_SCRIPT}"
fi

exec "${PYTHON_BIN}" "${OBJECTIVE_SCRIPT}" "$@"
