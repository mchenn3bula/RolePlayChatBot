#!/usr/bin/env bash
# Run as root in a dedicated Ubuntu 24.04 WSL distribution.
set -euo pipefail
source /etc/os-release
[[ "${ID}:${VERSION_ID}" == "ubuntu:24.04" ]] || { echo 'Requires Ubuntu 24.04.'; exit 1; }
[[ ${EUID} -eq 0 && -e /dev/dxg ]] || { echo 'Run as root inside WSL2 with /dev/dxg.'; exit 1; }
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3-venv python3-pip ca-certificates curl
mkdir -p /var/cache/roleplay-setup
cd /var/cache/roleplay-setup
curl -fL --retry 3 -o amdgpu-install.deb https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
apt-get install -y ./amdgpu-install.deb
apt-get update
# User-space ROCm only. WSL uses the installed Windows GPU driver.
apt-get install -y rocm
curl -fL --retry 3 -o rocdxg-roct.deb https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
apt-get install -y ./rocdxg-roct.deb
ldconfig
id -u n3bula >/dev/null 2>&1 || useradd --create-home --shell /bin/bash n3bula
usermod -a -G video,render n3bula
HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo
