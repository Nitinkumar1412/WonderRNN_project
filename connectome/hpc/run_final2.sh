#!/bin/bash
# Final lean members with the distillation settings the fold-0 sweep favoured (more teacher, 12 epochs):
#   fin_l0   distill.lam=0.0   (teacher only)
#   fin_l25  distill.lam=0.25
# Both on ALL data (fold -1), lean shape, init from pre_lean. The fold-0 gain may be partly teacher leakage
# (teachers saw fold-0 labels via the other folds), so confirm on the platform before trusting it.
#   cd ~/wunder_fund_challenge/code/connectome && bash hpc/run_final2.sh
# Then export each alone:
#   sbatch hpc/submit_cpu.sh export --run fin_l0  export.name=single_l0
#   sbatch hpc/submit_cpu.sh export --run fin_l25 export.name=single_l25
set -euo pipefail
# shellcheck disable=SC1090
source "$(dirname "${BASH_SOURCE[0]}")/_config.sh"
cd "$CODE_DIR"
M=(model.kind=lrnn model.d_proj=96 model.lrnn_layers=3 model.lrnn_state=320 model.lrnn_mlp=128 train.batch=192
   distill.teacher=zoo1 train.init_from=pre_lean train.epochs=12)
j=$(sbatch --parsable --job-name=wnn_fin_l0 hpc/submit_gpu.sh train --run fin_l0 --fold -1 "${M[@]}" distill.lam=0.0 seed=20261301)
echo "fin_l0 -> job $j"
j=$(sbatch --parsable --job-name=wnn_fin_l25 hpc/submit_gpu.sh train --run fin_l25 --fold -1 "${M[@]}" distill.lam=0.25 seed=20261302)
echo "fin_l25 -> job $j"
