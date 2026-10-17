# Literature and contest review (as of 2026-10-17)

Scope: what is publicly known that could move the Alfa Connectome score. Evidence quality is marked:
**[read]** = I read the source text, **[abstract]** = only the abstract/search snippet was available (claims
unverified), **[none]** = searched, found nothing usable.

## 1. This competition

**[read] dozken/wunder, challenge-3 notes** (https://github.com/dozken/wunder, `challenge-3/README.md`, `CLAUDE.md`) —
the only detailed public write-up. Another competitor's log, public ~0.657 at best:
- GRU 192×2 + 64–128 input projection + first-difference inputs; AdamW/cosine; EMA/SWA; train + non-held-out valid.
- Biggest lever: **sequence-level (global) weighted Pearson with running statistics** (per-chunk Pearson hurt).
- Loss masked to likely-scored rows with a mask predictor (held-out AUC 0.94). Oracle test: mask worth ≈ **+0.04**.
- **State reset every 2000–4000 rows** wins under the global metric; 2000 ≈ 4000; other periods no better.
- Ensembles: raw mean of 4 checkpoints +0.0025 (within noise, SE ≈ 0.0017). **Pairing models trained with different
  losses helped; same-loss pairs added nothing.**
- No gain from width/depth at equal params (GRU 384×2: +0.003), LSTM worse, aux is-scored head no help, per-sequence
  recentering very costly, soft state refresh no help.
- Local → public offset −0.030 to −0.037. **Scorer speed drifts day to day** (same graph 38/52/62 min; a two-graph
  ensemble killed at 69m59s). Official limit text there: 60 min, hard cap 70.
- Their own ideas: distil the ensemble into one graph; exploit loss diversity; better mask predictor; spend headroom
  on context/features rather than width.

**[none]** no winning write-up exists yet. The organisers say submissions are rescored on the validation dataset
between Nov 15 and Dec 1, 2026 (https://wunder-challenge.io/). Check the rules on whether that file is also
what we train on.

## 2. Recurrent architectures that run at constant cost per step

| Idea | Source | Evidence | Fit here |
|---|---|---|---|
| Gate depends on the input only → parallel-scan training, elementwise state (minGRU/minLSTM) | Feng et al., "Were RNNs All We Needed?", arXiv 2410.01201 | [abstract] | **Already used** (`lrnn`). Competitive with GRU in our runs. |
| Negative eigenvalues in the transition → better state tracking (Mamba, DeltaNet) | Grazzi et al., ICLR 2025, arXiv 2411.12537 | [abstract] | Our decay `1-z` is in (0,1): positive only. Allowing (-1,1) is a small kernel change. Gain on this data unknown. |
| Gated delta rule: gating (erase) + delta rule (targeted write), matrix state | Gated DeltaNet, arXiv 2412.06464 (ICLR 2025) | [abstract] | Matrix state costs more per step on one CPU; not obviously inside our ~35 µs budget. Low priority. |
| RG-LRU in Griffin, RWKV-7, xLSTM | arXiv 2402.19427, 2503.14456 | [abstract/none] | No CPU latency numbers found; speculative. |
| Financial benchmark: temporal-representation models (VSN+LSTM, xLSTM) beat generic deep models | arXiv 2603.01820 (daily futures) | [abstract] | Different data (daily, cross-asset); little transfer. |

Take-away: our lrnn already sits in the family the literature recommends; architectural novelty is unlikely to
close a 0.017 gap by itself. The competition evidence says loss, mask and data handling matter more.

## 3. Objectives and targets

- **Label Horizon Paradox**, arXiv 2602.03395 [abstract]: the best training label is often not the evaluation target
  but an intermediate-horizon proxy, because of a signal/noise trade-off; proposes bi-level proxy-label selection.
  Fit: we already train aux heads on y(t+10), y(t+50) (training only). A direct test is cheap: train on a mix of y
  and smoothed future y, or tune the aux horizons/weights on held-out WP. Targets here have lag-1 autocorrelation
  0.96, so expect modest effects.
- Pearson/IC as a loss: surveys describe IC as the metric but I found **[none]** on differentiable-Pearson training
  tricks beyond what the competition write-up reports; our implementation (running moments + exact micro-batch
  gradient) already matches the metric.

## 4. Distillation and ensembling

- Ensemble knowledge distillation (e.g. arXiv 2204.00548 [abstract]); "essence" distillation argues that parts of an
  ensemble's soft targets are noisy and should be down-weighted or mixed with hard labels (Yang et al. 2019
  [abstract]). All are classification-oriented; **[none]** for low-SNR regression.
- Fit: our `lam = 0.5` mix of y and teacher output is the standard recipe. Untested knobs: `lam`, using teachers only
  where they agree, and **teachers trained with different losses** (supported by the competition evidence in §1).

## 5. Selection bias / hidden scoring mask

- Importance weighting under covariate shift (arXiv 1910.06324, 2602.06713 [abstract]): weights = p_test/p_train;
  caution: high variance, smaller effective sample size. Our `floor + (1-floor)·P(scored)` weighting is the same idea.
- **Our own finding (from the logs):** the mask model finished at BCE 0.25 versus 0.30 for the constant base rate
  (8.6% scored); an AUC-0.94 model should be near 0.15, and we never measured AUC. The mask is the largest
  documented lever (+0.04 oracle) and currently the weakest component.

## 6. Test-time adaptation

- FSNet (ICLR 2023), TAFAS (AAAI 2025), PETSA (2025), norm-only TTA (arXiv 2602.00073), Test-Time Learning
  (arXiv 2409.14012) [abstract]. These need ground truth or gradient steps at test time.
- Not usable as published: this scorer shows features only (no targets) and we have one CPU core and a tight
  time limit. A cheap, label-free cousin would be input features normalised by a slow running mean/variance
  (regime-adaptive scaling); an lrnn can already learn this internally, so expect little.

## 7. Kaggle-style finance contests

Searched Jane Street (Real-Time Market Data), Ubiquant, DRW Crypto, G-Research Crypto: **[none]** — no winner
write-ups retrievable through search. Not used.

## 8. Ranked plan from this review

1. Fix the mask model (AUC measured on held-out valid sequences, target ≥ 0.94); relabel; retrain students. (§1, §5)
2. Teachers with different losses; distil the students from the mix. (§1, §4)
3. Train final models on all data; keep ensemble optional given scorer drift. (§1)
4. Cheap ablations: aux-horizon / smoothed-label mix (§3); negative-eigenvalue decay (§2).
5. Not worth it now: Gated DeltaNet / matrix-state models, test-time adaptation, wider models.

Honest limits: many entries are abstract-only, no paper reports results on data like ours, and the evidence for step 1–3
comes from one other competitor's log. Treat every item as a hypothesis to test on held-out folds.
