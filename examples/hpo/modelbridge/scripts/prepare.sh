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
CONFIG_PATH="$(resolve_path "${CONFIG_FILE:-config/config.yaml}")"

echo "Running prepare.sh..."
echo "Config file: ${CONFIG_PATH}"
echo "Workspace: ${WORKSPACE_PATH}"

mkdir -p "${WORKSPACE_PATH}/commands"
mkdir -p "${WORKSPACE_PATH}/logs"
mkdir -p "${WORKSPACE_PATH}/models"
mkdir -p "${WORKSPACE_PATH}/pairs"
mkdir -p "${WORKSPACE_PATH}/runs"
mkdir -p "${WORKSPACE_PATH}/state"

"${PYTHON:-python}" "${PROJECT_ROOT}/aiaccel/hpo/modelbridge/prepare.py" \
    --config "${CONFIG_PATH}" \
    --workspace "${WORKSPACE_PATH}"

echo "Prepare script finished."
