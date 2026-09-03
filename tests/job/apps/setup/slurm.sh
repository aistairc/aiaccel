#!/usr/bin/env bash

set -euxo pipefail

# Install
sudo apt-get update
sudo apt-get install -y \
  munge \
  slurm-wlm

# Configure
HOSTNAME="$(hostname -s)"
NODE_CONFIG="$(slurmd -C | head -n 1)"

sudo cp \
  "$(dirname "$0")/slurm.conf" \
  /etc/slurm/slurm.conf

sudo sed -i \
  -e "s/^SlurmctldHost=.*/SlurmctldHost=${HOSTNAME}/" \
  -e "s/^NodeName=.*/${NODE_CONFIG}/" \
  -e "s/Nodes=slurm/Nodes=${HOSTNAME}/" \
  /etc/slurm/slurm.conf

sudo mkdir -p \
  /var/spool/slurmctld \
  /var/spool/slurmd \
  /var/log/slurm

sudo chown slurm:slurm \
  /var/spool/slurmctld \
  /var/log/slurm

sudo chown root:root \
  /var/spool/slurmd

# Start
sudo systemctl restart munge
sudo systemctl restart slurmctld
sudo systemctl restart slurmd

# Wait
for _ in $(seq 1 30); do
  if sinfo >/dev/null 2>&1; then
    exit 0
  fi

  sleep 1
done

# failure diagnostics
sudo journalctl -u slurmctld
sudo journalctl -u slurmd
exit 1