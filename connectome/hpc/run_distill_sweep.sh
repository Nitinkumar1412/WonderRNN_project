#!/bin/bash
# Distillation knobs on fold 0 (lean lrnn students, same seed as div_a, which scored 0.6781; div_a2 0.6765).
#   sw_l25   distill.lam=0.25   (more teacher, less truth)
#   sw_l75   distill.lam=0.75   (less teacher)
#   sw_l00   distill.lam=0.0    (teacher only)
#   sw_e12   train.epochs=12    (lam 0.5, twice the training)
# Seed noise is ~0.0016, so only a gain above ~0.003 counts. Needs pre_lean and zoo1.
#   cd ~/wunder_fund_challenge/code/connectome && bash hpc/run_distill_sweep.sh
#   grep -h "final held-out" logs/wnn_sw_*.out | grep -o "wp_valid_mask': [0-9.]*"   (order: l25 l75 l00 e12 by name)
set -euo pipefail
# shellcheck disable=SC1090
source "$(dirname "${BASH_SOURCE[0]}")/_config.sh"
cd "$CODE_DIR"
M=(model.kind=lrnn model.d_proj=96 model.lrnn_layers=3 model.lrnn_state=320 model.lrnn_mlp=128 train.batch=192
   distill.teacher=zoo1 distill.lam=0.5 train.init_from=pre_lean seed=20261201)
run() { # name, overrides...
    local n="$1"; shift
    j=$(sbatch --parsable --job-name="wnn_sw_$n" hpc/submit_gpu.sh train --run "sw_$n" --fold 0 "${M[@]}" "$@")
    echo "sw_$n -> job $j"
}
run l25 distill.lam=0.25
run l75 distill.lam=0.75
run l00 distill.lam=0.0
run e12 train.epochs=12
