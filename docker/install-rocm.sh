#!/usr/bin/env bash
set -xeuo pipefail

wget https://repo.radeon.com/amdgpu-install/22.20.3/ubuntu/focal/amdgpu-install_22.20.50203-1_all.deb
apt-get install -y ./amdgpu-install_22.20.50203-1_all.deb
amdgpu-install -y --usecase=hiplibsdk,rocm --no-dkms
