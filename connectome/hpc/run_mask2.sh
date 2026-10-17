#!/bin/bash
# Stronger mask model. The mask only weights TRAINING rows (never shipped), so it may read future rows:
#   mask_bi      bidirectional GRU over whole 20k sequences      (expected to be the better one)
#   mask_causal  same size, causal                                (reference: how much does the future help?)
# Both hold out fold 0 of the valid sequences and report AUC after every epoch (target: >= 0.94).
# Then label-mask relabels every sequence with mask_bi (writes arrays/mask_p.f16.npy).
#   cd ~/wunder_fund_challenge/code/connectome && bash hpc/run_mask2.sh
#   grep -h "final held-out" logs/wnn_mask_bi_*.out logs/wnn_mask_causal_*.out
# After this, retrain the students (hpc/run_final.sh) so they use the new weights.
set -euo pipefail
# shellcheck disable=SC1090
source "$(dirname "${BASH_SOURCE[0]}")/_config.sh"
cd "$CODE_DIR"
COMMON=(train.objective=mask model.kind=gru model.d_proj=96 model.gru_hidden=128 model.gru_layers=2
        model.reset_every=0 model.aux_horizons=[] train.batch=24 train.epochs=40 train.lr=1.0e-3
        train.warmup_steps=50 train.eval_max_seqs=0 train.ema=0)
b=$(sbatch --parsable --job-name=wnn_mask_bi hpc/submit_gpu.sh train --run mask_bi --fold 0 "${COMMON[@]}" model.bidirectional=true)
c=$(sbatch --parsable --job-name=wnn_mask_causal hpc/submit_gpu.sh train --run mask_causal --fold 0 "${COMMON[@]}")
l=$(sbatch --parsable --job-name=wnn_label2 --dependency="afterok:$b" hpc/submit_gpu.sh label-mask --run mask_bi)
echo "mask_bi $b, mask_causal $c, relabel $l (after mask_bi)"
