FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.pixi/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        munge \
        slurm-wlm \
    && curl -fsSL https://pixi.sh/install.sh | bash \
    && rm -rf /var/lib/apt/lists/*

COPY slurm.conf /etc/slurm/slurm.conf
COPY cgroup.conf /etc/slurm/cgroup.conf
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh \
    && mkdir -p \
        /var/spool/slurmctld \
        /var/spool/slurmd \
        /var/log/slurm \
    && chown slurm:slurm \
        /var/spool/slurmctld \
        /var/log/slurm \
    && chown root:root \
        /var/spool/slurmd

ENTRYPOINT ["/entrypoint.sh"]
