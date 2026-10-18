#!/bin/bash
# Platform check of how much the teachers really help (fold-0 distillation numbers are biased by teacher leakage:
# single_lean lam=0.5 -> 0.65433, single_l25 -> 0.65393, single_l0 -> 0.65197).
#   fin_nod   no distillation at all (plain targets)
#   fin_l75   distill.lam=0.75
# Same lean shape, all data, 6 epochs, init pre_lean, like fin_lean_s1.
#   cd ~/wunder_fund_challenge/code/connectome && bash hpc/run_final3.sh
#   sbatch hpc/submit_cpu.sh export --run fin_nod export.name=single_nod
#   sbatch hpc/submit_cpu.sh export --run fin_l75 export.name=single_l75
set -euo pipefail
# shellcheck disable=SC1090
source "$(dirname "${BASH_SOURCE[0]}")/_config.sh"
cd "$CODE_DIR"
M=(model.kind=lrnn model.d_proj=96 model.lrnn_layers=3 model.lrnn_state=320 model.lrnn_mlp=128 train.batch=192
   train.init_from=pre_lean)
j=$(sbatch --parsable --job-name=wnn_fin_nod hpc/submit_gpu.sh train --run fin_nod --fold -1 "${M[@]}" distill.teacher= seed=20261401)
echo "fin_nod -> job $j"
j=$(sbatch --parsable --job-name=wnn_fin_l75 hpc/submit_gpu.sh train --run fin_l75 --fold -1 "${M[@]}" distill.teacher=zoo1 distill.lam=0.75 seed=20261402)
echo "fin_l75 -> job $j"
