#!/bin/bash
# One-factor sweep on a held-out fold: one GPU job per value, everything else from the base.
#   bash hpc/sweep.sh model.reset_every 1000 2000 4000 0
#   bash hpc/sweep.sh loss.moment_ema 0 0.5 0.9
#   BASE="model.kind=lrnn model.d_proj=96 model.lrnn_state=384" FOLD=1 bash hpc/sweep.sh model.lrnn_mlp 0 128 256
#   TAG=lr bash hpc/sweep.sh train.lr 2e-4 5e-4 8e-4
# Runs are named sw_<TAG or key>_<value>_f<FOLD>; compare with
#   bash hpc/compare.sh sw_
# reset_every=0 trains on whole 20k-row sequences: lower train.batch (BASE="train.batch=24").
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source hpc/_config.sh
KEY="${1:?usage: bash hpc/sweep.sh <config.key> <value> [value ...]}"; shift
FOLD="${FOLD:-0}"
TAG="${TAG:-$(echo "$KEY" | tr '.' '_')}"
read -r -a base <<< "${BASE:-}"
extra_sbatch=()
[ -n "${WNN_SBATCH_ARGS:-}" ] && read -r -a extra_sbatch <<< "$WNN_SBATCH_ARGS"
for v in "$@"; do
    run="sw_${TAG}_$(echo "$v" | tr -c 'A-Za-z0-9.\n-' '_')_f${FOLD}"
    id=$(sbatch --parsable --job-name="wnn_$run" ${extra_sbatch[@]+"${extra_sbatch[@]}"} hpc/submit_gpu.sh \
         train --run "$run" --fold "$FOLD" ${base[@]+"${base[@]}"} "$KEY=$v")
    echo "$run  job $id"
done
