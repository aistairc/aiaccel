#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

resolve_path() {
    local path_value="$1"
    if [[ "${path_value}" = /* ]]; then
        printf '%s\n' "${path_value}"
    else
        printf '%s\n' "${EXAMPLE_DIR}/${path_value}"
    fi
}

WORKSPACE_PATH="$(resolve_path "${WORKSPACE_DIR:-workspace}")"

echo "Running evaluate.sh..."

"${PYTHON:-python}" "${PROJECT_ROOT}/aiaccel/hpo/modelbridge/evaluate.py" --workspace "${WORKSPACE_PATH}"

echo "evaluate.sh finished."
