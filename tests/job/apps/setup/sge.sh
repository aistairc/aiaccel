#!/usr/bin/env bash

set -euxo pipefail

# Install
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  libdb5.3-dev \
  libhwloc-dev \
  libncurses-dev \
  libpam0g-dev \
  libssl-dev \
  libsystemd-dev \
  libtirpc-dev \
  libxext-dev \
  pkgconf

# Build
git clone --depth 1 \
  https://github.com/daimh/sge.git \
  /tmp/sge

cd /tmp/sge

cmake -S . -B build \
  -DCMAKE_INSTALL_PREFIX=/opt/sge

cmake --build build -j"$(nproc)"
sudo cmake --install build

# Configure
SGE_HOSTNAME="sge"
IP_ADDRESS="$(hostname -I | awk '{print $1}')"

sudo hostname "${SGE_HOSTNAME}"

sudo sed -i \
  "/${IP_ADDRESS}/d" \
  /etc/hosts

echo "${IP_ADDRESS} ${SGE_HOSTNAME}" \
  | sudo tee -a /etc/hosts

test "$(hostname -f)" = "${SGE_HOSTNAME}"

sudo useradd -r -d /opt/sge sge || true
sudo chown -R sge /opt/sge

cd /opt/sge

sudo sh -c 'yes "" | ./install_qmaster'
sudo sh -c 'yes "" | ./install_execd'

sudo env \
  SGE_ROOT=/opt/sge \
  SGE_CELL=default \
  /opt/sge/bin/lx-amd64/qconf \
  -as "$(hostname -f)"

echo "/opt/sge/bin/lx-amd64" >> "$GITHUB_PATH"
echo "SGE_ROOT=/opt/sge" >> "$GITHUB_ENV"
echo "SGE_CELL=default" >> "$GITHUB_ENV"
