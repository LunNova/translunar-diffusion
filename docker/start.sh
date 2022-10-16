#!/usr/bin/env bash

# USAGE:
# ./start.sh [cuda/rocm] [force]
# Defaults to cuda
# If force set, rebuilds even if tag already exists

# FIXME: docker support? This is currently podman only

set -xeuo pipefail

type=${1:-cuda}

tag=translunar-diffusion-$type:latest
container=translunar-diffusion-$type-container

echo "Using tag $tag"

if [ "$type" = "rocm" ]; then
  devices="--device /dev/kfd --device /dev/dri"
else
  # FIXME: pass through more than just 0
  devices="--device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm"
fi

cp ../diffusion/scripts/preload_models.py preload.py

if ! podman inspect --type=image $tag || [[ "z${2:-}" = "zforce" ]]; then
  podman build -f $type.Dockerfile --format docker -t $tag .
fi
podman rm $container || true
podman run --name $container --rm -it -v $(realpath ../):/content/ \
  -v /mnt/scratch/derpibooru/:/inputs/derpibooru/:ro \
 --userns=keep-id --ipc=host \
  -u 1000:1000 \
  $devices \
  --security-opt seccomp=unconfined --group-add video \
  --network host \
  --log-driver none \
  $tag \
  bash -c ". /usr/local/bin/_activate_current_env.sh && id && cd /content && ls -l . && exec bash" || true
podman rm $container || true
