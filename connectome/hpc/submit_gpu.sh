#!/bin/bash
# GPU step (train / label-mask / teacher / evaluate / soup):
#   cd ~/wunder_fund_challenge/code/connectome
#   sbatch hpc/submit_gpu.sh train --run gru_r2k_f0 --fold 0
#   sbatch hpc/submit_gpu.sh train --run big_lrnn_f1 --fold 1 model.kind=lrnn model.d_proj=256 model.lrnn_state=1024
#   sbatch hpc/submit_gpu.sh teacher --runs big_gru_f0,big_gru_f1 distill.teacher=zoo1
#   sbatch hpc/submit_gpu.sh evaluate --run stu_f0
#
# Development node: one H100 NVL (93 GB) split into 94 shards (~1 GB each). The default
# shard:24 lets four training runs share the GPU (a GRU 192x2 run with batch 256 x 2000 rows
# peaks near 10 GB; a big lrnn teacher can need ~40 GB -> --gres=shard:47). Whole GPU:
#   sbatch --gres=gpu:1 hpc/submit_gpu.sh ...        (other clusters: -p gpu --gres=gpu:1)
# On CUDA OOM lower train.batch / eval.batch.

#SBATCH --job-name=wnn_gpu
#SBATCH --gres=shard:24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
_cfg=""
for _d in "${SLURM_SUBMIT_DIR:-.}/hpc" "${WNN_CODE_DIR:-$HOME/wunder_fund_challenge/code/connectome}/hpc"; do
    [ -f "$_d/_config.sh" ] && { _cfg="$_d/_config.sh"; break; }
done
[ -n "$_cfg" ] || { echo "ERROR: hpc/_config.sh not found - sbatch from code/connectome or set WNN_CODE_DIR" >&2; exit 1; }
# shellcheck disable=SC1090
source "$_cfg"
activate_env
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

STEP="${1:?usage: sbatch hpc/submit_gpu.sh <step> [--run R] [--fold F] [key=value ...]}"; shift
echo "[wnn] node=$(hostname) step=$STEP args=$* gpu=${CUDA_VISIBLE_DEVICES:-?} work=${WNN_WORK_DIR:-$WNN_ROOT/work}"
# dataloader workers follow the CPU allocation
python -m wnn "$STEP" "$@" "train.workers=$(( ${SLURM_CPUS_PER_TASK:-12} - 2 ))"
