#!/bin/bash
# CPU step (prepare / export / preflight):
#   cd ~/wunder_fund_challenge/code/connectome
#   sbatch hpc/submit_cpu.sh prepare
#   sbatch hpc/submit_cpu.sh export --run stu_f0 export.name=sub_v1
# Sized for the development node (128 CPUs, 125 GB RAM, default queue). prepare reads the
# 35 GB of parquet with up to 32 processes and writes ~60 GB of arrays (float16).

#SBATCH --job-name=wnn_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
_cfg=""
for _d in "${SLURM_SUBMIT_DIR:-.}/hpc" "${WNN_CODE_DIR:-$HOME/wunder_fund_challenge/code/connectome}/hpc"; do
    [ -f "$_d/_config.sh" ] && { _cfg="$_d/_config.sh"; break; }
done
[ -n "$_cfg" ] || { echo "ERROR: hpc/_config.sh not found - sbatch from code/connectome or set WNN_CODE_DIR" >&2; exit 1; }
# shellcheck disable=SC1090
source "$_cfg"
activate_env

STEP="${1:?usage: sbatch hpc/submit_cpu.sh <step> [--run R] [key=value ...]}"; shift
echo "[wnn] node=$(hostname) step=$STEP args=$* cpus=${SLURM_CPUS_PER_TASK:-?} work=${WNN_WORK_DIR:-$WNN_ROOT/work}"
python -m wnn "$STEP" "$@"
