#!/usr/bin/env bash

# USAGE:
# ./start.sh [cuda/rocm] [force]
# Defaults to cuda
# If force set, rebuilds even if tag already exists

set -xeuo pipefail

type=${1:-cuda}

tag=lun-diffusion-$type:latest
container=lun-diffusion-$type-container

cp ../../scripts/preload_models.py preload.py

if ! podman inspect --type=image $tag || [[ "z${2:-}" = "zforce" ]]; then
  podman build -f $type.Dockerfile --format docker -t $tag .
fi
podman rm $container || true
podman run --name $container --rm -it -v $(realpath ../../):/content/ \
  -v /mnt/scratch/derpibooru/:/inputs/derpibooru/:ro \
 --userns=keep-id --ipc=host \
  -u 1000:1000 \
  --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --group-add video \
  --network host \
  $tag \
  bash -c ". /usr/local/bin/_activate_current_env.sh && id && cd /content && ls -l . && pip install --user -e . && exec bash" || true
podman rm $container || true
