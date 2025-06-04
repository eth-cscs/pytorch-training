#!/bin/bash
set -euo pipefail

# Always run from the script's directory
cd "$(dirname "$0")"

# Detect container engine
if command -v docker &>/dev/null; then
  OCIRUN="docker"
elif command -v podman &>/dev/null; then
  OCIRUN="podman"
else
  echo "Error: podman or docker not found" >&2
  exit 1
fi

# Project directories
SLIDES_DIR="$PWD/src"
EXERCISES_DIR="$PWD/../notebooks"

## Jupyter Lab configuration file
#JUPYTER_CONFIG="$(pwd)/jupyter_lab_config.py"

# Container image name
IMAGE_NAME="slidev-pytorch"

# Container user
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Common volume mounts
VOLUMES=(
  -v "$SLIDES_DIR":/slides:Z
  -v "$EXERCISES_DIR":/exercises:Z
  #-v "$JUPYTER_CONFIG":/root/.jupyter/jupyter_lab_config.py:Z
)

# Common options
COMMON_OPTS=(
  --rm -ti
  -w /slides
  -p 3030:3030
  -p 8888:8888
  -p 9000:9000
  "${VOLUMES[@]}"
)

# Command to run inside the container
CMD='
find /exercises -name "*.ipynb" -exec jupyter trust {} \; >/dev/null 2>&1 && \
cd /exercises && jupyter lab --ip=0.0.0.0 --no-browser \
    --IdentityProvider.token="" \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.tornado_settings="{\"headers\": {\"Content-Security-Policy\": \"frame-ancestors *\", \"X-Frame-Options\": \"ALLOWALL\"}}" \
    --ServerApp.terminals_enabled=False \
    --ServerApp.log_level=WARN \
    --ServerApp.websocket_ping_interval=40000 \
    --ServerApp.websocket_ping_timeout=30000 \
    --ServerApp.allow_origin='*' & \
python3 -m http.server 9000 --directory /slides & \
cd /slides && pnpm install && pnpm dev --remote -o false
'

# Run the container
if [ "$OCIRUN" = "podman" ]; then
  $OCIRUN run "${COMMON_OPTS[@]}" \
    --userns=keep-id \
    --user "$USER_ID:$GROUP_ID" \
    "$IMAGE_NAME" bash -c "$CMD"
else
  $OCIRUN run "${COMMON_OPTS[@]}" \
    --user "$USER_ID:$GROUP_ID" \
    "$IMAGE_NAME" bash -c "$CMD"
fi
