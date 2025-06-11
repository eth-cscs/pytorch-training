#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

command -v podman &>/dev/null && ocirun="podman"
command -v docker &>/dev/null && ocirun="docker"

$ocirun build -t slidev-pytorch -f Containerfile .

