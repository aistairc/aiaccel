#!/usr/bin/env bash
set -euo pipefail

image="${SLURM_IMAGE:-slurm-ci}"
container="${SLURM_CONTAINER:-slurm}"

cleanup() {
    docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker build \
    -t "$image" \
    ci/slurm

docker run \
    --name "$container" \
    --hostname slurm \
    --privileged \
    --cgroupns=private \
    -d \
    "$image"

echo "Waiting for Slurm..."

for _ in $(seq 1 30); do
    if docker exec "$container" sinfo >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

docker exec "$container" sinfo

docker exec "$container" \
    sbatch \
    --wait \
    --output=/tmp/job.out \
    --wrap='echo hello'

actual="$(docker exec "$container" cat /tmp/job.out)"

if [[ "$actual" != "hello" ]]; then
    echo "unexpected output: $actual" >&2
    exit 1
fi

echo "Slurm smoke test: OK"
