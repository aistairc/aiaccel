#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

image="${SLURM_IMAGE:-slurm-ci}"
container="${SLURM_CONTAINER:-slurm}"

cleanup() {
    docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> Building Slurm image"
docker build \
    -f "$script_dir/Dockerfile" \
    -t "$image" \
    "$script_dir"

echo "==> Starting Slurm container"
docker run \
    --name "$container" \
    --hostname slurm \
    --privileged \
    --cgroupns=private \
    -d \
    "$image"

echo "==> Waiting for Slurm"

ready=0

for _ in $(seq 1 30); do
    if docker exec "$container" sinfo >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done

if [[ "$ready" != 1 ]]; then
    echo "Slurm did not become ready" >&2
    docker logs "$container" >&2 || true
    exit 1
fi

docker exec "$container" sinfo

echo "==> Running smoke job"

docker exec "$container" rm -f /tmp/job.out

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
