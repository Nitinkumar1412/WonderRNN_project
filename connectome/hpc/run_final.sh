#!/bin/bash
# Final ensemble members: lrnn students trained on ALL sequences (no held-out fold), distilled from the
# out-of-fold zoo, different seeds. Run on the login node after run_pipeline.sh finished (needs zoo1, pre_lrnn):
#   cd ~/wunder_fund_challenge/code/connectome && bash hpc/run_final.sh [N_SEEDS] [SHAPE]
# SHAPE: full (d96 S384 x3 mlp192, 464k MACs, 34 us/row) | lean (d96 S320 x3 mlp128, ~340k MACs)
# lean needs its own pretrain (shapes differ), full reuses pre_lrnn. Then export the ensemble:
#   sbatch hpc/submit_cpu.sh export --run fin_full_s1,fin_full_s2 export.name=ens2_full
set -euo pipefail
# shellcheck disable=SC1090
source "$(dirname "${BASH_SOURCE[0]}")/_config.sh"
cd "$CODE_DIR"
N="${1:-2}"; SHAPE="${2:-full}"
case "$SHAPE" in
  full) M=(model.kind=lrnn model.d_proj=96 model.lrnn_layers=3 model.lrnn_state=384 model.lrnn_mlp=192 train.batch=192); PRE_RUN=pre_lrnn; dep="" ;;
  lean) M=(model.kind=lrnn model.d_proj=96 model.lrnn_layers=3 model.lrnn_state=320 model.lrnn_mlp=128 train.batch=192); PRE_RUN=pre_lean
        dep=$(sbatch --parsable --job-name=wnn_pre_lean hpc/submit_gpu.sh train --run pre_lean --fold -1 "${M[@]}" \
              train.objective=pretrain model.aux_horizons=[] train.epochs=3 loss.mask_source=none)
        dep="--dependency=afterok:$dep" ;;
  *) echo "SHAPE must be full|lean" >&2; exit 1 ;;
esac
for i in $(seq 1 "$N"); do
    id=$(sbatch --parsable --job-name="wnn_fin_${SHAPE}_s$i" $dep hpc/submit_gpu.sh train --run "fin_${SHAPE}_s$i" --fold -1 \
         "${M[@]}" distill.teacher=zoo1 distill.lam=0.5 "train.init_from=$PRE_RUN" "seed=$((20261100 + i))")
    echo "fin_${SHAPE}_s$i -> job $id"
done
