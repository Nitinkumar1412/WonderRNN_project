#!/bin/bash
# Wider distilled students that use the latency headroom (platform ran 464k MACs in 21m47s; limit 60 min).
# A: lrnn d64 S1024 x4 mlp128 (866k MACs, 64 us/row local ~ 42 min)   B: GRU 256x2 (717k, 48 us ~ 31 min)
# Per shape: pretrain, then a distilled fold-0 student. Run on the login node:
#   cd ~/wunder_fund_challenge/code/connectome && bash hpc/run_wide.sh
# Compare with the 464k student's 0.6765:  grep -h "final held-out" logs/wnn_stu_w?_f0_*.out
set -euo pipefail
_cfg="$(dirname "${BASH_SOURCE[0]}")/_config.sh"
# shellcheck disable=SC1090
source "$_cfg"
cd "$CODE_DIR"

A=(model.kind=lrnn model.d_proj=64 model.lrnn_layers=4 model.lrnn_state=1024 model.lrnn_mlp=128 train.batch=128)
B=(model.kind=gru model.d_proj=128 model.gru_hidden=256 model.gru_layers=2 train.batch=192)
PRE=(train.objective=pretrain model.aux_horizons=[] train.epochs=3 loss.mask_source=none)
DIST=(distill.teacher=zoo1 distill.lam=0.5)

for tag in A B; do
    declare -n shape="$tag"
    t=$(echo "$tag" | tr 'A-Z' 'a-z')
    p=$(sbatch --parsable --job-name="wnn_pre_w$t" hpc/submit_gpu.sh train --run "pre_w$t" --fold -1 "${shape[@]}" "${PRE[@]}")
    s=$(sbatch --parsable --job-name="wnn_stu_w${t}_f0" --dependency="afterok:$p" hpc/submit_gpu.sh \
        train --run "stu_w${t}_f0" --fold 0 "${shape[@]}" "${DIST[@]}" "train.init_from=pre_w$t")
    echo "shape $tag: pretrain $p -> student $s"
done
