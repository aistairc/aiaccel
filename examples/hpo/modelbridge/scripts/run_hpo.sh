#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

resolve_path() {
    local path_value="$1"
    if [[ "${path_value}" = /* ]]; then
        printf '%s\n' "${path_value}"
    else
        printf '%s\n' "${EXAMPLE_DIR}/${path_value}"
    fi
}

PHASE="${1:?phase must be train or test}"
WORKSPACE_PATH="$(resolve_path "${WORKSPACE_DIR:-workspace}")"
RUNS_DIR="${WORKSPACE_PATH}/runs/${PHASE}"

echo "Running run_hpo.sh for phase: ${PHASE}..."
if [ ! -d "${RUNS_DIR}" ]; then
    echo "Error: directory not found: ${RUNS_DIR}"
    exit 1
fi

find "${RUNS_DIR}" -name "config.yaml" -print0 | sort -z | while IFS= read -r -d '' CONFIG_PATH; do
    WORK_DIR="$(dirname "${CONFIG_PATH}")"
    echo "-----------------------------------------"
    echo "  [HPO] Starting optimization in: ${WORK_DIR}"
    echo "  [HPO] Config: ${CONFIG_PATH}"
    aiaccel-hpo optimize --config "${CONFIG_PATH}"
done

echo "run_hpo.sh for phase ${PHASE} finished."
