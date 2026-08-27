#!/usr/bin/env bash

set -euo pipefail

#
# Prepare cgroup v2 hierarchy.
#
# Keep the container root cgroup empty so controllers can be
# delegated to child cgroups.
#
mkdir -p /sys/fs/cgroup/daemons

# Move PID 1 (this entrypoint) into a child cgroup.
echo $$ > /sys/fs/cgroup/daemons/cgroup.procs

# Enable controllers for child cgroups.
echo "+cpuset +cpu +memory" \
    > /sys/fs/cgroup/cgroup.subtree_control

#
# Prepare Munge.
#
mkdir -p \
    /run/munge \
    /var/lib/munge \
    /var/log/munge

chown -R munge:munge \
    /run/munge \
    /var/lib/munge \
    /var/log/munge

chmod 755 /run/munge
chmod 711 /var/lib/munge
chmod 700 /var/log/munge

chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key

#
# Prepare Slurm directories.
#
mkdir -p \
    /var/spool/slurmctld \
    /var/spool/slurmd \
    /var/log/slurm

chown slurm:slurm \
    /var/spool/slurmctld \
    /var/log/slurm

chown root:root \
    /var/spool/slurmd

#
# Start daemons.
#
runuser -u munge -- munged

slurmctld

#
# IgnoreSystemd=yes expects this hierarchy.
#
mkdir -p /sys/fs/cgroup/system.slice

#
# IMPORTANT:
# Do not use "exec slurmd -D".
#
# Keep entrypoint.sh as PID 1.
#
slurmd -D &
slurmd_pid=$!

wait "$slurmd_pid"