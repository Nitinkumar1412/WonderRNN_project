#!/bin/bash
# Run a submission inside the scorer replica with 1 CPU: C vs numpy parity, us/row, WP.
#   bash tools/docker_check.sh <submission dir> [n_valid_seqs]
# Needs Docker and the starter pack (WNN_STARTERPACK, default <root>/wnn_connectome_starterpack).
# Local us/row is a lower bound: calibrate the scorer's slowdown with one real submission.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sub="$(cd "${1:?usage: tools/docker_check.sh <submission dir> [n]}" && pwd)"
n="${2:-2}"
sp="${WNN_STARTERPACK:-$(cd "$here/../.." && pwd)/wnn_connectome_starterpack}"
cp "$sp/requirements.txt" "$here/docker/requirements.txt"
docker image inspect wnn-scorer >/dev/null 2>&1 || docker build -t wnn-scorer -f "$here/docker/Dockerfile" "$here/docker"
# MSYS_NO_PATHCONV: keep Git-Bash on Windows from rewriting the container paths
MSYS_NO_PATHCONV=1 docker run --rm --cpus=1 --memory=16g --network=none \
    -v "$sub:/sub:ro" -v "$sp:/sp:ro" -v "$here/tools:/tools:ro" \
    wnn-scorer python /tools/check_submission.py /sub /sp/datasets/valid.parquet /sp "$n"
