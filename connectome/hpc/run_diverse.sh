#!/bin/bash
# Loss diversity test on fold 0 (lean lrnn students, same data/teacher/seed family, different losses):
#   div_a  default loss                              (mse 0.05, moment_ema 0.9, mask floor 0.1)
#   div_b  MSE-heavy, batch-local Pearson            (mse 0.5,  moment_ema 0,   mask floor 0.1)
#   div_c  default loss, no mask weighting           (mask_floor 1.0 -> every row weight |y|)
#   div_a2 default loss again, other seed            (control: same-loss pair)
# Then compares the held-out WP of the pairs:  a+b, a+c  versus  a+a2.
# Needs pre_lean (run_final.sh lean trains it) and zoo1.
#   cd ~/wunder_fund_challenge/code/connectome && bash hpc/run_diverse.sh
#   grep -h "ensemble_wp" logs/wnn_div_eval_*.out
set -euo pipefail
# shellcheck disable=SC1090
source "$(dirname "${BASH_SOURCE[0]}")/_config.sh"
cd "$CODE_DIR"
M=(model.kind=lrnn model.d_proj=96 model.lrnn_layers=3 model.lrnn_state=320 model.lrnn_mlp=128 train.batch=192
   distill.teacher=zoo1 distill.lam=0.5 train.init_from=pre_lean)
declare -A X
X[a]="seed=20261201"
X[b]="seed=20261202 loss.mse_weight=0.5 loss.moment_ema=0"
X[c]="seed=20261203 loss.mask_floor=1.0"
X[a2]="seed=20261204"
ids=""
for k in a b c a2; do
    # shellcheck disable=SC2086
    j=$(sbatch --parsable --job-name="wnn_div_$k" hpc/submit_gpu.sh train --run "div_$k" --fold 0 "${M[@]}" ${X[$k]})
    echo "div_$k -> job $j"; ids="$ids:$j"
done
for pair in a,b a,c a,a2; do
    sbatch --parsable --job-name=wnn_div_eval --dependency="afterok${ids}" hpc/submit_gpu.sh evaluate --runs "div_${pair%,*},div_${pair#*,}" >/dev/null
done
echo "pair evaluations queued (after all four finish)"
