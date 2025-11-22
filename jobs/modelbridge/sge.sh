#!/bin/bash
#$ -cwd
#$ -l s_rt=${JOB_TIME:-02:00:00}
#$ -l s_vmem=${JOB_MEM:-8G}
#$ -q ${JOB_QUEUE:-all.q}
#$ -j y
#$ -o ${MODELBRIDGE_OUTPUT:-./work/modelbridge}/logs/sge_${PHASE:-phase}_${ROLE:-role}_${TARGET:-target}_${RUN_ID:-all}.log

# SGE execution helper for modelbridge phases. Can be invoked directly (bash) or via qsub.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <log_path> -- <command> [args...]" >&2
  exit 1
fi

LOG_PATH="$1"
shift
if [[ "${1:-}" == "--" ]]; then
  shift
fi

cmd=("$@")

WORKDIR="${SGE_O_WORKDIR:-${WORKDIR:-$PWD}}"
cd "${WORKDIR}"
mkdir -p "$(dirname "$LOG_PATH")"

ROOT="${ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
export MODELBRIDGE_CONFIG MODELBRIDGE_OUTPUT MODELBRIDGE_SCENARIO TRAIN_RUN_IDS EVAL_RUN_IDS

LOG_DIR="${MODELBRIDGE_OUTPUT:-./work/modelbridge}/logs"
mkdir -p "$LOG_DIR"

exec "${cmd[@]}" >"$LOG_PATH" 2>&1
