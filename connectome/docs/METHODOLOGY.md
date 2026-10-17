# Alfa Connectome — methodology, techniques and results

Competition: Wunder Fund "Alfa Connectome". Predict `t0`, `t1` for every row of 20,000-row
sequences from 112 features, one row per `predict()` call.
Metric: global weighted Pearson (weights = |y|, values clipped to ±2) on a hidden mask of rows.
Scorer: 1 vCPU, 60 minutes for ~39.4M rows (~90 µs/row including the harness), zip ≤ 20 MB, Python 3.11.

## 1. Result so far

| Step | Held-out WP (fold 0, valid sequences, real mask) |
|---|---|
| Starter baseline GRU (public leaderboard) | ~0.617 |
| Safe GRU 192×2, global-Pearson loss, resets | 0.6618 |
| Big teachers (GRU 512×3 4.4M params, lrnn 1024-state 6.4M params) | 0.655–0.694 across folds, ≈ the small GRU on the same fold |
| Distilled students (hybrid, pretrained) | 0.6744 / 0.6749 / 0.6743 |
| **Distilled lrnn student (d96, state 384, 3 blocks)** | **0.6765** |
| Weight soup of three hybrid students | 0.6375 (failed, see §6) |

**Platform:** the distilled lrnn student scored **0.65370** on the platform and ran in
**21m47s** (615 MB). Locally the same model projected to 22.6 min, so the scorer is about as fast
as the 1-CPU Docker replica (factor ≈ 0.96) and the C kernel compiled there. The local→platform
score gap is ≈ 0.023 (0.677 → 0.654). So latency is *not* yet binding: the model uses about a
third of the 60-minute budget.

(Correction: an earlier draft of this doc quoted 63m55s from a misread result and concluded the
scorer was 2.8× slower. The 21m47s figure is the real one.)

## 2. Pipeline

```
parquet ─ prepare ─ memmap (f16) + stats + 5 folds (hash of seq_ix)
   ├─ mask model        P(is_scored) per row, trained on valid sequences only → soft row weights
   ├─ pretrain          self-supervised next-dx prediction → backbone init
   ├─ teacher zoo       big GRU + big linear-RNN, one per fold → out-of-fold soft targets
   ├─ students          small GRU / lrnn / hybrid, distilled from the zoo, init from pretrain
   └─ export            model.npz (f16 matrices) + solution.py + kernel.c → zip
```

Everything is driven by `configs/default.yaml` with `key=value` overrides, and run on Slurm
(`hpc/`). `hpc/run_pipeline.sh extreme` chains the phases; `hpc/run_small.sh` trains the
smaller students for the latency limit.

## 3. Techniques and why

**Streaming-compatible models.** One row per call, so the model must be a recurrence with O(1)
cost per step.
- `gru`: `[x, dx]` → Linear+GELU → GRU layers → head.
- `lrnn`: minGRU-style input-gated linear recurrence, `h = (1-z)h + z·g(b)`, where gates depend
  only on the input. This trains with a parallel scan (chunked, log-space `logcumsumexp`) on the
  H100, and inference is elementwise in the state, so a wide state is cheap.
- `hybrid`: GRU layer(s) followed by lrnn blocks.

**Inputs.** Standardised features clipped to ±10, plus first differences `dx_t = x_t − x_{t-1}`.
Inputs are rounded to float16 exactly as the training store, so the scorer sees the same values.

**Periodic state reset** (`reset_every = 2000` rows), in both training and inference. Hidden
state quality degrades late in a sequence under the global metric; resetting fixes that, and
makes every 2000-row segment an independent training sample, so training is fully parallel.

**Loss = the metric.** Global weighted Pearson pooled over the whole batch, with running moments
across batches (`moment_ema`), plus a small weighted MSE term to anchor scale. Outputs are
bounded to `2·tanh(y/2)`, so the loss sees exactly what the metric clips. Warm-up rows 0–98 run
through the model but are excluded from the loss.

**Mask-aware row weights.** Only ~10% of rows are scored. A small mask model predicts
`P(is_scored)` (AUC ≈ 0.94 in public notes); rows are weighted by
`|y|·(floor + (1−floor)·P)`. The real mask is used on valid sequences.

**Auxiliary heads.** Heads predicting `y_{t+10}`, `y_{t+50}` regularise training and are dropped
at export.

**Self-supervised pretraining.** Predicting the next `dx` gives a backbone initialisation.

**Teacher → student distillation.** Five folds × (big GRU, big lrnn) produce out-of-fold
predictions for every sequence; students train on `0.5·y + 0.5·teacher`. This is the largest
single gain here, about +0.012 to +0.015 over a plain GRU at identical inference cost.

**Optimisation.** AdamW, warm-up + cosine LR, bf16 autocast (scans in fp32), EMA weights
(saved and evaluated), gradient clipping.

**GPU-memory robustness (shared H100).** The cluster GPU is shared, and another process held
~86 GB. On CUDA OOM, training turns on activation checkpointing, then halves the micro-batch.
The global-Pearson gradient stays *exact* under micro-batching (two-pass: moments first, then
per-chunk gradient coefficients; unit-tested against the full batch). Evaluation halves its batch
and retries.

## 4. Inference engine

`inference/solution.py` + `inference/kernel.c`:
- Weights in `model.npz`, matrices stored float16 (halves the bytes read; WP changes by ~1e-4).
- The C kernel is compiled with `gcc -O3 -march=native` at start-up on the scorer, with
  register-blocked matvec, vectorised exp/sigmoid/tanh/GELU (8 lanes, GCC vector extensions) and
  F16C float16 loads. Falls back to numpy automatically (about 5× slower, would time out).
- Verified: numpy vs torch ≤ 1e-5, C vs numpy ≤ 1e-6, deterministic, identical WP.

## 5. Latency model (1 CPU Docker replica)

Per-row time ≈ **4 µs fixed + ~6 µs per 100k MACs/step**. One platform data point: the
464k-MAC student ran 21m47s against 22.6 min locally. Keep ≥ 30% margin under 60 min because
the scorer's speed may vary between runs (another competitor reported drift), i.e. aim for
≲ 40 min locally ≈ 60 µs/row ≈ 900k MACs.

| Model | MACs/step | Local µs/row | Local min | Platform min |
|---|---:|---:|---:|---:|
| submitted lrnn d96/S384/3 blocks | 464k | 34.4 | 22.6 | **21.8 (measured)** |
| GRU 256×2 | 717k | 47.8 | 31.4 | – |
| lrnn d64 S1024 ×4 | 866k | 64.5 | 42.4 | – |
| lrnn d128 S512 ×4 | 1078k | 79.7 | 52.3 | – (too close) |
| lrnn d96 S192 ×2, MLP96 | 169k | 13.5 | 8.9 | – |
| lrnn d64 S256 ×2, no MLP | 113k | 10.2 | 6.7 | – |

`tools/latency_grid.py` writes random-weight submissions per shape; `tools/docker_check.sh`
times them in a 1-CPU container.

## 6. What did not work

- **Weight soup** of three students from the same pretrain: 0.6375 vs ≈0.674 for each, because
  they land in different basins. Combine models by distillation or averaging predictions.
- **Bigger teachers alone** did not beat the small GRU; their value is as teachers.
- A **possible small optimism** in student scores: teacher targets for training folds came from
  models that had seen the evaluation fold's inputs and targets (not through the student's
  targets). Treat the fold-0 numbers as slightly high; the platform gap (0.677 → 0.654) is larger
  than that.

## 7. Next steps

1. Spend the latency headroom: train a wider student (≈ 700–900k MACs, e.g. lrnn d64/S1024 or
   a larger hybrid) distilled from the same zoo, and compare on fold 0 against 0.6765.
2. Final model on all folds, a few seeds, average predictions (or distil from the average).
3. Re-measure the platform time with each submission; keep ≥ 30% margin under 60 min.
4. Close the local→platform gap (0.023): check mask-weighting and reset period against the
   platform score, not only fold 0.
