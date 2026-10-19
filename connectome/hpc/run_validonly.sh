#!/bin/bash
# Is valid.parquet a better stand-in for the hidden test set than a random fold?
# Trains the lean lrnn WITHOUT teachers on the train.parquet sequences only (fold -2) and scores it on ALL
# valid.parquet sequences with the real mask. No distillation, so no teacher leakage.
# Compare with the platform score of the same recipe (single_nod = 0.65383, single_lean = 0.65433):
#   ~0.654  -> valid is a faithful proxy; use fold -2 for every model-selection decision from now on
#   ~0.677  -> the 0.023 platform gap is not a train/valid split effect (test is simply different / harder)
# Two seeds give the noise level.
#   cd ~/wunder_fund_challenge/code/connectome && bash hpc/run_validonly.sh
#   grep -h "final held-out" logs/wnn_vo_*.out | grep -o "wp_valid_mask': [0-9.]*"
set -euo pipefail
# shellcheck disable=SC1090
source "$(dirname "${BASH_SOURCE[0]}")/_config.sh"
cd "$CODE_DIR"
M=(model.kind=lrnn model.d_proj=96 model.lrnn_layers=3 model.lrnn_state=320 model.lrnn_mlp=128 train.batch=192
   train.init_from=pre_lean distill.teacher= train.eval_max_seqs=400)
for s in 1 2; do
    j=$(sbatch --parsable --job-name="wnn_vo_s$s" hpc/submit_gpu.sh train --run "vo_s$s" --fold -2 "${M[@]}" "seed=$((20261500 + s))")
    echo "vo_s$s -> job $j"
done
