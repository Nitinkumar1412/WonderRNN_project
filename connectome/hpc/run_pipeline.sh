#!/bin/bash
# Extreme track as a chain of dependent SLURM jobs. Run on the login node:
#
#   bash hpc/run_pipeline.sh smoke              # 60 sequences, a few steps per job - exercises every step
#   bash hpc/run_pipeline.sh extreme            # the real thing (all phases below)
#   bash hpc/run_pipeline.sh extreme zoo        # start from a phase (earlier outputs must exist)
#
# Phases (each waits for the previous one):
#   data      prepare: parquet -> arrays (CPU)
#   mask      mask model on valid sequences -> label-mask (P(scored) for every row)
#   base      SAFE GRU on fold 0 (the safety-net submission) + self-supervised pretraining of
#             both student backbones (lrnn, hybrid) on every sequence's features
#   zoo       teacher zoo: big GRU + big linear-recurrence model on each of the 5 folds (10 runs)
#             -> teacher: out-of-fold averaged soft targets  teachers/zoo1.f16.npy
#   students  fold 0: distilled GRU, lrnn and hybrid students (from the pretrained backbones),
#             plus 2 more seeds of the hybrid -> soup of the 3 hybrids
# Then compare fold-0 scores (runs/*/metrics.json, `python -m wnn evaluate --run ...`),
# train the winner on everything (--fold -1), export, and check latency in Docker.
#
# Watch:  squeue -u $USER ; tail -f logs/wnn_<name>_<jobid>.out
# Extra sbatch options for every job:  WNN_SBATCH_ARGS="--time=12:00:00" bash hpc/run_pipeline.sh extreme

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source hpc/_config.sh
MODE="${1:-extreme}"
FROM="${2:-data}"
PHASES=" data mask base zoo students "
case "$PHASES" in *" $FROM "*) ;; *) echo "unknown phase '$FROM' (one of:$PHASES)" >&2; exit 1 ;; esac
case "$MODE" in smoke|extreme) ;; *) echo "mode must be smoke|extreme" >&2; exit 1 ;; esac

# ---------------------------------------------------------------- model presets
# Students are sized to roughly the weight count of the safe GRU 192x2 (~450k per step):
# at batch 1 on one CPU the step time follows the bytes of weights read. Re-size after
# measuring with tools/docker_check.sh and one real submission.
SAFE=(model.kind=gru model.d_proj=128 model.gru_hidden=192 model.gru_layers=2)
STU_LRNN=(model.kind=lrnn model.d_proj=96 model.lrnn_layers=3 model.lrnn_state=384 model.lrnn_mlp=192 train.batch=192)
STU_HYB=(model.kind=hybrid model.d_proj=128 model.gru_hidden=128 model.gru_layers=1 model.lrnn_layers=2
         model.lrnn_state=384 model.lrnn_mlp=0 train.batch=192)
BIG_GRU=(model.kind=gru model.d_proj=256 model.gru_hidden=512 model.gru_layers=3 train.epochs=10 train.batch=192)
BIG_LRNN=(model.kind=lrnn model.d_proj=256 model.lrnn_layers=6 model.lrnn_state=1024 model.lrnn_mlp=512
          train.epochs=10 train.batch=48)
MASK=(train.objective=mask model.kind=gru model.d_proj=64 model.gru_hidden=96 model.gru_layers=1
      model.aux_horizons=[] train.epochs=8)
PRE=(train.objective=pretrain model.aux_horizons=[] train.epochs=3 loss.mask_source=none)
DIST=(distill.teacher=zoo1 distill.lam=0.5)
NF=5
BIG_SHARD="${BIG_SHARD:-47}"     # big teachers get half the H100 (shard:47); others shard:24

EXTRA=()
if [ "$MODE" = "smoke" ]; then
    export WNN_WORK_DIR="$WNN_ROOT/work_smoke"
    EXTRA=(train.max_steps=20 train.epochs=1 train.batch=16 train.eval_max_seqs=8 eval.max_seqs=8)
fi
PREP_EXTRA=(); [ "$MODE" = "smoke" ] && PREP_EXTRA=(data.max_seqs=60)

wnn_need_writable logs "${WNN_WORK_DIR:-$WNN_ROOT/work}" || exit 1
wnn_env_ready || bash hpc/setup_env.sh

extra_sbatch=()
[ -n "${WNN_SBATCH_ARGS:-}" ] && read -r -a extra_sbatch <<< "$WNN_SBATCH_ARGS"
submit() {  # submit <cpu|gpu> <job-name> <deps> <args...>
    local kind="$1" name="$2" deps="$3"; shift 3
    local opts=(--parsable --job-name="wnn_$name" ${extra_sbatch[@]+"${extra_sbatch[@]}"})
    [ -n "$deps" ] && opts+=(--dependency="afterok:$deps")
    [ "$kind" = gpu ] && [ -n "${BIG_SHARD:-}" ] && [[ "$name" == big* ]] && opts+=(--gres="shard:$BIG_SHARD")
    sbatch "${opts[@]}" "hpc/submit_$kind.sh" "$@"
}
join() { local IFS=:; echo "$*"; }

started=0
phase() { if [ "$started" = 1 ] || [ "$1" = "$FROM" ]; then started=1; return 0; fi; return 1; }
last=""

if phase data; then
    last=$(submit cpu prepare "$last" prepare ${PREP_EXTRA[@]+"${PREP_EXTRA[@]}"}); echo "data      $last"
fi
if phase mask; then
    t=$(submit gpu mask "$last" train --run mask_v1 --fold -1 "${MASK[@]}" ${EXTRA[@]+"${EXTRA[@]}"})
    last=$(submit gpu label "$t" label-mask --run mask_v1); echo "mask      $t -> $last"
fi
if phase base; then
    ids=()
    ids+=("$(submit gpu safe_f0 "$last" train --run safe_gru_f0 --fold 0 "${SAFE[@]}" ${EXTRA[@]+"${EXTRA[@]}"})")
    ids+=("$(submit gpu pre_lrnn "$last" train --run pre_lrnn --fold -1 "${STU_LRNN[@]}" "${PRE[@]}" ${EXTRA[@]+"${EXTRA[@]}"})")
    ids+=("$(submit gpu pre_hyb "$last" train --run pre_hyb --fold -1 "${STU_HYB[@]}" "${PRE[@]}" ${EXTRA[@]+"${EXTRA[@]}"})")
    last=$(join "${ids[@]}"); echo "base      $last"
fi
if phase zoo; then
    ids=()
    for f in $(seq 0 $((NF - 1))); do
        ids+=("$(submit gpu "big_gru_f$f" "$last" train --run "big_gru_f$f" --fold "$f" "${BIG_GRU[@]}" ${EXTRA[@]+"${EXTRA[@]}"})")
        ids+=("$(submit gpu "big_lrnn_f$f" "$last" train --run "big_lrnn_f$f" --fold "$f" "${BIG_LRNN[@]}" ${EXTRA[@]+"${EXTRA[@]}"})")
    done
    runs=$(for f in $(seq 0 $((NF - 1))); do printf 'big_gru_f%s,big_lrnn_f%s,' "$f" "$f"; done)
    last=$(submit gpu teacher "$(join "${ids[@]}")" teacher --runs "${runs%,}" distill.teacher=zoo1); echo "zoo       $last"
fi
if phase students; then
    ids=()
    ids+=("$(submit gpu stu_gru_f0 "$last" train --run stu_gru_f0 --fold 0 "${SAFE[@]}" "${DIST[@]}" ${EXTRA[@]+"${EXTRA[@]}"})")
    ids+=("$(submit gpu stu_lrnn_f0 "$last" train --run stu_lrnn_f0 --fold 0 "${STU_LRNN[@]}" "${DIST[@]}" train.init_from=pre_lrnn ${EXTRA[@]+"${EXTRA[@]}"})")
    hyb=()
    for s in 1 2 3; do
        hyb+=("$(submit gpu "stu_hyb_s${s}_f0" "$last" train --run "stu_hyb_s${s}_f0" --fold 0 "${STU_HYB[@]}" "${DIST[@]}" \
                 train.init_from=pre_hyb "seed=$((20261016 + s))" ${EXTRA[@]+"${EXTRA[@]}"})")
    done
    soup=$(submit gpu soup_hyb "$(join "${hyb[@]}")" soup --runs stu_hyb_s1_f0,stu_hyb_s2_f0,stu_hyb_s3_f0 --run soup_hyb_f0)
    last=$(join "${ids[@]}" "$soup"); echo "students  $last"
fi
echo "submitted ($MODE). Final job(s): ${last:-none}"
echo "Compare:  grep -h '\"wp_valid_mask\"' ${WNN_WORK_DIR:-$WNN_ROOT/work}/runs/*/metrics.json   (or python -m wnn evaluate --run NAME)"
