# Alfa Connectome — extreme track

Full methodology, results and latency model: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

Predict `t0`, `t1` row by row on one CPU core; metric = global weighted Pearson.
The extreme track trains big models on the H100 and ships one small recurrent
network that runs in a hand-written C kernel compiled on the scoring machine.

```
                 ┌─ mask model ── P(is_scored) per row ──────────────┐ (loss weights)
parquet ─ prepare┤                                                   │
 (memmap f16)    ├─ pretrain ─── self-supervised next-dx backbones ──┤ (init)
                 ├─ teacher zoo ─ big GRU + big linear-RNN × 5 folds ─┤ out-of-fold soft targets
                 │                                                   ▼
                 └─────────────────────────────── students (GRU / lrnn / hybrid) ── soup
                                                                     │
                       export ── model.npz (f16 matrices) + solution.py + kernel.c ── zip
```

**Models** (`src/wnn/models.py`, all O(1) per step, state reset every `reset_every` rows):
- `gru`: `[x, dx]` → Linear+GELU → GRU layers → head.
- `lrnn`: input-gated linear recurrence blocks (minGRU-style, `h = (1-z)h + z·g(b)`), trained
  with a chunked log-space parallel scan. Elementwise state, so a wide state is cheap.
- `hybrid`: GRU layer(s), then lrnn blocks.

The outputs are bounded to `2·tanh(y/2)`, so the loss sees exactly what the metric clips.

**Loss**: the global weighted Pearson computed over the whole batch, with an optional
running-moments version (`loss.moment_ema`), plus a small weighted MSE term. Rows are
weighted by `|y|·(floor + (1-floor)·P(scored))`. Heads that predict future targets are
used during training only (`model.aux_horizons`) and are not exported.

## Guarantees (tested)
- Training forward equals the step-by-step inference to under 1e-4 for gru/lrnn/hybrid, with
  resets, f32 and f16 weights (`tests/test_core.py`).
- Inputs are rounded to float16 exactly as the training store was, so a real valid sequence
  replayed from parquet matches torch to about 1e-6.
- C kernel and numpy agree to under 1e-6 in the scorer replica, and the C kernel is deterministic.

## Measured latency (local Docker replica, 1 CPU, this laptop; scorer limit 60 min)

| model | kMACs/step | f32 µs/row | f16 µs/row | min / 39.4M rows (f16) |
|---|---:|---:|---:|---:|
| baseline GRU 128×2 | 226 | 16 | 17 | 11 |
| safe GRU 192×2 | 435 | 40 | 29 | 19 |
| student hybrid (d128) | 422 | 47 | 30 | 20 |
| student lrnn (d96, S384) | 464 | 55 | 33 | 21 |
| GRU 256×2 | 717 | 89 | 48 | 32 |
| lrnn d64 S1024 ×4 | 866 | 87 | 67 | 44 |

A fixed overhead of about 7 µs/row is the scorer's Python loop. **The scorer is slower than a
local Docker run** (another competitor saw about 3× on a Mac), so submit the safe GRU first and
calibrate: *platform minutes ÷ local projected minutes*. Then size the students with
`tools/latency_grid.py`.

## Run on the cluster (Slurm, same layout as Amazon_ML)

```bash
# upload <root>/code/connectome and <root>/wnn_connectome_starterpack (datasets/ + utils.py)
cd <root>/code/connectome
bash hpc/setup_env.sh                 # once, on the login node (internet)
bash hpc/preflight.sh                 # data, packages, unit tests
bash hpc/run_pipeline.sh smoke        # 60 sequences, every phase, ~1 h
bash hpc/run_pipeline.sh extreme      # data -> mask -> base -> zoo -> students
bash hpc/run_pipeline.sh extreme zoo  # resume from a phase
bash hpc/compare.sh                   # held-out table, best first
```

Single steps: `sbatch hpc/submit_gpu.sh train --run NAME --fold 0 key=value ...`,
`sbatch hpc/submit_cpu.sh export --run NAME`. One-factor sweeps:
`bash hpc/sweep.sh model.reset_every 1000 2000 4000` (then `bash hpc/compare.sh sw_`).
GPU jobs default to `--gres=shard:24` (four share the H100); the big teachers use shard:47.
Intermediate files go to `/scratch/$USER/wnn_work` when it exists, otherwise `<root>/work`.
`train` resumes from `runs/<run>/last.pt` when a job is re-submitted.

## Steps (`python -m wnn <step>`, config `configs/default.yaml`, any key overridable)

| step | does |
|---|---|
| prepare | parquet → `arrays/X.f16.npy [N,20000,112]`, `Y`, `scored`, stats, 5 hashed folds |
| train | one model on one fold; `train.objective` = target / mask / pretrain; `train.init_from`, `distill.teacher` |
| label-mask | mask run → `arrays/mask_p.f16.npy` |
| teacher | `--runs a,b,...` → out-of-fold averaged soft targets `teachers/<name>.f16.npy` |
| soup | weight-average same-architecture runs → a new run, scored against its ingredients |
| evaluate | held-out WP with the real valid mask, all rows, and by quarter |
| export | zip + parity vs torch + latency bench; `export.weights=f16` by default |

Before uploading: `bash tools/docker_check.sh <work>/submissions/<name> 2`. It builds the
scorer replica (`docker/Dockerfile`) and reports C/numpy µs/row, WP and C-vs-numpy agreement.

## Open questions to settle early
1. **Compiling C at start-up.** The scorer image has `build-essential`, and `solution.py` falls
   back to numpy if gcc fails. Ask the organisers (Discord) to confirm that compiling the
   bundled `kernel.c` is allowed.
2. **Platform speed factor.** Calibrate it with the first real submission.
3. Numpy fallback speed is about 3–5× slower than C, so a model that fits only with C must not
   be shipped until (1) is confirmed.
