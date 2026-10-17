#!/bin/bash
# Smaller distilled students for the platform's time limit (the 464k-MAC student took 63m55s there).
# Per shape: self-supervised pretrain, then a distilled fold-0 student. Run on the login node:
#   cd ~/wunder_fund_challenge/code/connectome && bash hpc/run_small.sh
# Then compare with:  bash hpc/compare.sh   (or grep "final held-out" logs/wnn_stu_s*_f0_*.out)
# Local us/row (1 CPU Docker): A 13.5, B 10.2   (stu_lrnn d96/s384/x3 was 34.4)
set -euo pipefail
_cfg="$(dirname "${BASH_SOURCE[0]}")/_config.sh"
# shellcheck disable=SC1090
source "$_cfg"
cd "$CODE_DIR"

A=(model.kind=lrnn model.d_proj=96 model.lrnn_layers=2 model.lrnn_state=192 model.lrnn_mlp=96 train.batch=192)
B=(model.kind=lrnn model.d_proj=64 model.lrnn_layers=2 model.lrnn_state=256 model.lrnn_mlp=0 train.batch=192)
PRE=(train.objective=pretrain model.aux_horizons=[] train.epochs=3 loss.mask_source=none)
DIST=(distill.teacher=zoo1 distill.lam=0.5)

for tag in A B; do
    declare -n shape="$tag"
    t=$(echo "$tag" | tr 'A-Z' 'a-z')
    p=$(sbatch --parsable --job-name="wnn_pre_s$t" hpc/submit_gpu.sh train --run "pre_s$t" --fold -1 "${shape[@]}" "${PRE[@]}")
    s=$(sbatch --parsable --job-name="wnn_stu_s${t}_f0" --dependency="afterok:$p" hpc/submit_gpu.sh \
        train --run "stu_s${t}_f0" --fold 0 "${shape[@]}" "${DIST[@]}" "train.init_from=pre_s$t")
    echo "shape $tag: pretrain $p -> student $s"
done
