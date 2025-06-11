#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v podman &>/dev/null; then
   OCIRUN="podman"
elif command -v docker &>/dev/null; then
  OCIRUN="docker"
else
  echo "Error: podman or docker not found" >&2
  exit 1
fi

$OCIRUN build -t slidev-pytorch -f Containerfile .
