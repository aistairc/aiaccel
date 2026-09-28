#!/usr/bin/env bash

set -euxo pipefail

# Install
sudo apt-get update
sudo apt-get install -y \
  git \
  build-essential \
  m4 \
  autoconf \
  automake \
  libtool \
  libhwloc-dev \
  libx11-dev \
  libxt-dev \
  libedit-dev \
  libical-dev \
  libncurses-dev \
  libpq-dev \
  python3-dev \
  tcl-dev \
  tk-dev \
  swig \
  libexpat1-dev \
  libssl-dev \
  libxext-dev \
  libxft-dev \
  libcjson-dev \
  pkg-config \
  expat \
  postgresql \
  perl \
  sendmail-bin

# Build
git clone --depth 1 \
  https://github.com/openpbs/openpbs.git \
  /tmp/openpbs

cd /tmp/openpbs

./autogen.sh
PYTHON=/usr/bin/python3 ./configure --prefix=/opt/pbs

make -j"$(nproc)"
sudo make install

sudo /opt/pbs/libexec/pbs_postinstall

# Configure PBS
sudo sed -i \
  's/^PBS_START_MOM=.*/PBS_START_MOM=1/' \
  /etc/pbs.conf

sudo chmod 4755 \
  /opt/pbs/sbin/pbs_iff \
  /opt/pbs/sbin/pbs_rcp

# Start
sudo /opt/pbs/libexec/pbs_init.d start

# Wait
ready=false

for _ in $(seq 1 30); do
  if /opt/pbs/bin/qstat -B >/dev/null 2>&1; then
    ready=true
    break
  fi

  sleep 1
done

# failure diagnostics
if [ "${ready}" != true ]; then
  sudo cat /var/spool/pbs/server_logs/* || true
  sudo cat /var/spool/pbs/mom_logs/* || true
  exit 1
fi

echo "/opt/pbs/bin" >> "$GITHUB_PATH"
echo "/opt/pbs/sbin" >> "$GITHUB_PATH"
