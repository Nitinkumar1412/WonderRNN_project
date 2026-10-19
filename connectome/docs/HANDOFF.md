# Alfa Connectome — team handoff (as of 2026-10-19)

Read this first. It is the single entry point for a new teammate: the problem, what the code does, how to
run it, everything we measured (including what failed and what is still unverified), and what to try next.
Companion docs: `METHODOLOGY.md` (techniques, latency model), `RESEARCH.md` (literature review, evidence-tagged).

**Status:** platform score ≈ **0.654** (rank ≈ 449; top 10 are 0.6696–0.6724). Final ranking is on a hidden
Final Dataset (rescoring Nov 15 – Dec 1, 2026). Deadline Nov 15, 2026.

---------------------------------------------------------------------------------------------------------

## 1. Problem

- Predict `t0`, `t1` for every row of sequences of 20,000 rows from 112 anonymised features. Rows 0–98 are
  warm-up (not scored).
- **Streaming:** the scorer calls `predict()` once per row, in order. The model must be causal and carry its own
  state. No future rows, no targets at test time.
- **Metric:** global weighted Pearson, `rho_k = WP(clip(y,±2), clip(pred,±2); w=|y|)` pooled over *all scored
  rows of the dataset*, averaged over `t0`, `t1`. Scored rows are a hidden mask, about 9% of rows, in sticky runs.
- **Scorer:** 1 vCPU, 16 GB, **60 min** for ~39.4M rows (~90 µs/row all-in), zip ≤ 20 MB, Python 3.11
  slim-bookworm (has build-essential, so gcc works), no internet, 5 submissions/day.
- **Data facts:** features are rank-transformed (~±2.3; `a*` up to ±5); targets std ~0.95, 22% exact zeros
  (weight 0), `t0`/`t1` correlation −0.74, lag-1 autocorrelation 0.96. The scored mask is predictable from
  `a2,a3,a4`. Difficulty is regime-driven (`a5`, `a7`).
- **Rules we follow:** no external data; training on `valid.parquet` is allowed; repo stays private; choose the
  final submission by held-out CV and timing safety, not by the public leaderboard (it is noisy).

## 2. Repo map (`code/connectome/`)

| Path | What |
|---|---|
| `configs/default.yaml` | every knob; override with `key=value` on the CLI |
| `src/wnn/` | `data` (store, folds), `prepare`, `models` (gru/lrnn/hybrid), `metric` (exact scorer + torch loss), `train`, `mask`, `teacher`, `evaluate`, `export`, `soup`, `cli` |
| `inference/solution.py`, `kernel.c` | the shipped submission code (numpy fallback + C kernel, ensemble support) |
| `hpc/*.sh` | Slurm launchers (`run_pipeline.sh`, `run_final*.sh`, `run_mask2.sh`, `run_diverse.sh`, `run_distill_sweep.sh`, `submit_gpu.sh`, `submit_cpu.sh`) |
| `tools/` | `latency_grid.py` (random-weight timing grid), `docker_check.sh` (1-CPU replica), `check_submission.py` (parity + determinism) |
| `tests/test_core.py` | 13 tests: metric, exact micro-batch gradient, scan, torch-vs-solution for every kind, ensemble, bidirectional guard |

CLI: `python -m wnn <step> [--fold F] [--run NAME] [--runs a,b] key=value ...`
Steps: `preflight prepare train label-mask teacher soup evaluate export`.
`evaluate --runs A,B` reports each member and the mean-of-outputs WP on the held-out fold.

## 3. Environment and workflow

- Dev PC (Windows, `.venv-dev`): unit tests, Docker timing, submissions. No gcc locally; the C kernel is tested
  in Docker or on the HPC.
- HPC: `user3@172.17.120.214`, Slurm, one shared H100 NVL (`--gres=shard:N`), work dir
  `/var/tmp/user3/wnn_work` (arrays, runs, teachers, submissions).
- Git: private repo `Nitinkumar1412/WonderRNN_project`, branch `extreme-track-connectome`, files under `connectome/`.

**Pitfalls we hit (save yourself the time)**
1. Files written on Windows have CRLF; `sbatch` rejects them. After every upload run
   `sed -i 's/\r$//' hpc/*.sh src/wnn/*.py ...`.
2. `scp -r src configs hpc tests` into `~/wunder_fund_challenge/code/connectome/`. Do not scp single files to the
   project root.
3. Resumed checkpoints must restore the running Pearson moments to the right device (fixed in
   `metric.GlobalPearsonLoss.total`).
4. The shared GPU can be nearly full. Training auto-enables activation checkpointing then halves the micro-batch
   on OOM; the global-Pearson gradient stays exact (unit-tested).
5. `prepare` stores float16 arrays (~56 GB). A float32 rebuild needs its own `WNN_WORK_DIR`; we measured float16
   input rounding at ~0.1% of the `dx` std, so it is not worth it.

## 4. Pipeline (what each step does)

```
parquet -> prepare -> memmap f16 + feature stats + 5 folds (hash of seq_ix)
  mask model  (train on valid seqs)  -> label-mask -> soft P(is_scored) for every row
  pretrain    (next-dx, self-supervised) -> backbone init
  teacher zoo (big GRU 512x3 + big lrnn S1024 x6, one per fold) -> out-of-fold soft targets (zoo1)
  students    (small lrnn) trained on  lam*y + (1-lam)*teacher, init from pretrain
  export      model.npz (f16) + solution.py + kernel.c -> zip, parity check, latency bench
```
Typical commands: `bash hpc/run_pipeline.sh extreme` (whole chain), `bash hpc/run_final.sh 1 lean` (all-data
member), `sbatch hpc/submit_cpu.sh export --run fin_lean_s1 export.name=single_lean`, then download the zip,
`bash tools/docker_check.sh <dir> 1`, `python tools/check_submission.py <zip>`.

## 5. Techniques in the shipped model

- **Inputs:** standardised features clipped ±10, plus first difference `dx`; rounded to float16 exactly like the
  training store.
- **Loss = the metric:** global weighted Pearson over the whole batch with running moments (`moment_ema=0.9`),
  exact two-pass micro-batching, plus a small weighted MSE (0.05). Outputs bounded to `2*tanh(y/2)`.
  Warm-up rows run through the model but are excluded from the loss.
- **State reset every 2000 rows** in both training and inference. Makes each 2000-row segment an independent
  sample (fully parallel training) and fixes late-sequence state decay.
- **Mask-weighted rows:** weight `|y| * (floor + (1-floor)*P(scored))`, floor 0.1; the true mask on valid seqs.
- **Aux heads** for `y(t+10)`, `y(t+50)` (training only), **EMA** 0.999, AdamW + warm-up + cosine, bf16
  (scans in fp32), self-supervised pretraining.
- **Distillation** from the teacher zoo (`distill.lam` = weight on the true target).

## 6. Model architecture

Shipped model (`single_lean`: ~0.38M params, 372k MACs/step):
```
x_t (112) -> clip ±10 -> concat [x_t, dx_t] (224)
  -> Linear 96 + GELU
  -> 3 x linear-RNN block:   gates z and candidate b from the INPUT only
        h_t = (1 - z) * h_{t-1} + z * g(b)        state width 320 (elementwise)
        + MLP 128
  -> Linear 2 -> 2*tanh(y/2)
state zeroed every 2000 rows
```
Why: the gates do not depend on `h`, so training uses a chunked log-space parallel scan on the GPU, and
inference per step is elementwise on the state, so a wide state is cheap on one core.
Other kinds in `models.py`: `gru` (the baseline family) and `hybrid` (GRU layer(s) then lrnn blocks).
Training-only parts: big teachers (GRU 512x3 4.4M params; lrnn S1024 x6 6.4M), the mask model (small GRU),
aux heads.

## 7. Inference engine and latency

- `solution.py` loads `model.npz`; at start it compiles `kernel.c` (`gcc -O3 -march=native`): register-blocked
  matvec, vectorised exp/sigmoid/tanh/GELU, F16C float16 weight loads. Automatic numpy fallback (about 5x slower:
  would time out). Ensembles: `model_i.npz` members averaged (`AvgNet`).
- Verified: numpy vs torch ≤ 1e-5, C vs numpy ≤ 1e-6, bit-identical across two runs, identical WP.
- **Latency model (1-CPU Docker):** ≈ 4 µs fixed + 6 µs per 100k MACs per row.

| Model | MACs | Local µs/row | Platform |
|---|---:|---:|---|
| floor (d16, S16) | ~10k | 4.2 | – |
| `single_lean` d96 S320 x3 | 372k | ~27 | 16m46s |
| d96 S384 x3 | 464k | 34 | 21m47s |
| ens2_lean (2 members) | ~740k | ~60 | 33m51s |
| lrnn d64 S1024 x4 | 866k | 64 | 57m17s |
| lrnn d128 S512 x4 | 1078k | 80 | not run (too close) |

The platform speed drifts: ratio to the local projection was 0.96, ~0.8 and 1.35 on different days. Another
competitor saw 38/52/62 min for one graph and a killed 69m59s run. **Rule: local projection ≤ 35 min.**
float32 weights cost ~1.6x latency for ~1e-4 WP; keep float16.

## 8. Results (everything we measured)

**Platform (public leaderboard, noisy: treat < 0.002 as no difference)**

| Submission | Score | Time | Notes |
|---|---|---|---|
| `stu_lrnn_f0` | 0.65370 | 21m47s | d96 S384 x3, fold-0 trained (4/5 data), lam 0.5 |
| `stu_wa_f0` | 0.65334 | 57m17s | 2x compute; no gain |
| `ens2_lean` | 0.65456 | 33m51s | 2 members, new mask labels |
| `single_lean` | **0.65433** | **16m46s** | 1 member, all data, lam 0.5 — current fallback |
| `single_l25` | 0.65393 | 16m36s | lam 0.25, 12 epochs |
| `single_l0` | 0.65197 | 17m6s | lam 0.0 (teacher only), 12 epochs |
| `single_nod`, `single_l75` | **pending** | – | trained (`fin_nod`, `fin_l75`); export and submit |

**Held-out fold 0 (390 valid seqs, real mask; seed noise ≈ 0.0016)**

| Experiment | WP |
|---|---|
| Starter baseline (public) | ~0.617 |
| Safe GRU 192x2, global-Pearson loss, resets | 0.6618 |
| Hybrid students (pretrained, distilled) | 0.6743–0.6749 |
| Distilled lrnn d96 S384 x3 | 0.6765 |
| Wide student lrnn d64 S1024 x4 / GRU 256x2 | 0.6781 / 0.6682 |
| Lean lrnn, loss variants a / b / c / a2 | 0.6781 / 0.6780 / 0.6765 / 0.6765 |
| Pairs a+b, a+c, a+a2 (ensembles) | 0.6785 / 0.6781 / 0.6778 |
| lam 0.75 / 0.5 / 0.25 / 0.0 | 0.6775 / 0.6773 / 0.6801 / 0.6813 |
| 12 epochs (lam 0.5) | 0.6800 |
| Weight soup of 3 students | 0.6375 (failed) |
| Mask AUC: bidirectional / causal GRU / LightGBM on hand features | 0.892 / 0.901 / 0.829 |

## 9. Lessons (read before planning experiments)

1. **Fold-0 numbers involving distillation are biased upward.** Teacher outputs for the training folds came from
   models that saw fold 0's inputs and targets. The lam sweep looks monotone on fold 0 (more teacher = better) and
   is the *opposite* on the platform (lam 0.5 > 0.25 > 0.0). For anything touching the teachers, trust only
   platform scores, or build nested teachers (see §10.1).
2. **We never measured the undistilled student on the platform**, so the headline "+0.012 from distillation" is
   unverified. `single_nod` is the control; export and submit it.
3. **Plateau:** platform 0.652–0.655 across very different models (size, ensemble, mask, loss). Capacity is not
   the limit. The local→platform gap is ≈ 0.023 for every model, so judge changes by differences, not levels.
4. **Mask:** a better mask is not a free win here. AUC 0.90 is what the GRU reaches; dropping mask weighting
   changed nothing beyond noise. The other competitor's +0.04 was an *oracle* (perfect mask) result.
5. **Ensembles don't pay for their latency** (≤ +0.0004 on fold 0, +0.0002 on the platform).
6. **Width doesn't pay** (2x compute: +0.0016 on fold 0, −0.0004 and 57 min on the platform).
7. Single-seed noise (0.0016) is as big as most effects. Use ≥ 2 seeds or a gain threshold of ~0.003.
8. The public leaderboard has a few thousand scored rows' noise; do not tune to it.

## 10. What to try next (ranked, each a hypothesis)

### 10.1 Fix the evaluation first (cheap, high value)
- **Nested teachers for a leak-free fold-0:** train teachers on folds {1..4} only and score fold 0 students
  against them. Then fold-0 comparisons of `lam`, epochs, teacher type become trustworthy. Cost: ~5 GPU-hours.
- Run `single_nod` and `single_l75`. If `nod` ≈ 0.654, drop the teachers and spend the GPU time elsewhere.

### 10.2 New architectures (all must keep O(1) per step; budget ≈ 400–500k MACs)
1. **Negative / signed eigenvalues in the transition** (Grazzi et al., ICLR 2025, arXiv 2411.12537). Our decay
   `1-z` is in (0,1). Allow (−1,1), e.g. `a = 2*(1-z) - 1` or a separate signed gate. Tiny kernel change; better
   state tracking in theory. Gain on this data unknown.
2. **Multi-timescale state:** split the state into channels with fixed or learned slow/fast decay ranges
   (e.g. half with decay > 0.99). Regime signals (`a5`, `a7`) may live on slow timescales; the reset-2000 trick
   suggests long memory is partly useful.
3. **Complex / oscillatory diagonal recurrence** (LRU / S4D style, RG-LRU in Griffin arXiv 2402.19427). Two
   real lanes per complex state; cost about 2x per state element, but the elementwise path is cheap.
4. **Short causal convolution front end** (kernel 4–8 on the projected features) before the recurrence, as in
   Mamba/minGRU variants. Gives local temporal context for ~d*k MACs. Cheap to test, easy kernel change.
5. **Regime-gated mixture:** a tiny gate from slow features (`a5`, `a7` or a slow EMA of them) choosing among
   2–3 small expert heads. Targets the regime-driven difficulty noted in the data analysis.
6. **Gated DeltaNet / small matrix state** (arXiv 2412.06464): 2–4 heads of 16x16 state. Per-step cost is higher;
   probably needs d and layers cut. Lower priority, high ceiling.
7. **Output parametrisation exploiting corr(t0,t1) = −0.74:** predict a shared factor plus a residual per target.
8. Deeper but narrower, or a wider *input projection* with a thin state (the platform clock is dominated by
   matmuls, not the state).

### 10.3 Training and objective
- **Smoothed / intermediate-horizon labels** ("Label Horizon Paradox", arXiv 2602.03395): mix `y` with
  `mean(y[t..t+h])` for small `h`, or tune aux horizons/weights on held-out WP.
- **Reset period** 1000 / 4000 / staggered resets per layer (the other competitor found 2000 ≈ 4000).
- **SWA / checkpoint averaging** across the last epochs (soup of the same run, not of different runs).
- **Hyper-parameters never swept:** lr, weight decay, dropout/noise on inputs, batch size, `mask_floor`,
  `moment_ema`, `mse_weight`. A 2-seed random sweep on the lean shape costs ~17 min per run.
- **Data augmentation:** feature dropout, small gaussian noise on `x`, random start offsets within the 2000-row
  segment so reset boundaries move.
- **Cross-validated student selection:** pick the final recipe by mean over folds, not one fold.

### 10.4 Features
- Slow running statistics (EMA of each feature over ~200 and ~2000 rows) as extra inputs; costs one extra
  vector update per row. A label-free cousin of test-time adaptation.
- Rolling volatility of `dx`; cross-feature products of `a2,a3,a4` (the mask drivers).

### 10.5 Latency headroom (to afford more model)
- int8 weights with an integer matvec in the C kernel (the other competitor saw no gain for int8 GRU in ORT;
  in a hand kernel it may differ). Frees ~40% for width/depth.
- Fuse the three blocks' gate matmuls into one; share the input projection across blocks.

### 10.6 Probably not worth it (we or others tested)
Bigger teachers alone, weight soup across runs, ensembles of same-recipe members, wider students, bidirectional
mask, LSTM, per-sequence recentering, soft state refresh, test-time gradient adaptation (no labels at test).

## 11. Experiment protocol

1. State the hypothesis and the threshold (gain ≥ 0.003 on fold-0 mean of 2 seeds, or ≥ 0.002 on the platform).
2. Run on the lean shape (17 min on the H100) with fold 0, seed pair, same teacher and pretrain.
3. Distillation-related claims: platform-only until nested teachers exist.
4. Before any submission: Docker timing (`tools/docker_check.sh`), `tools/check_submission.py` (parity,
   determinism), local projection ≤ 35 min.
5. Record the result in §8 of this file.

## 12. Open items

- Export and submit `single_nod`, `single_l75` (decides whether distillation is worth keeping).
- Final-submission policy: the current best safe choice is `single_lean` (0.65433, 16m46s). Keep two final
  candidates (one fastest-safe, one best-score) and choose by held-out CV, not the public score.
- Keep the zip ≤ 20 MB (currently ~1.4 MB) and the repo private.
