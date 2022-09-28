#!/usr/bin/env bash
set -xeuo pipefail

apt-get install -y tzdata nvidia-cuda-dev # deepspeed needs nvidia cuda headers
