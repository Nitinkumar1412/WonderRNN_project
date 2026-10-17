"""CPU unit tests (no data files, no GPU):  python -m pytest tests -q

The key guarantee: training (parallel scan, cuDNN-style GRU, segment resets) and
inference (solution.py numpy step, step by step) compute the same function.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wnn.config import load_config                      # noqa: E402
from wnn.export import round_matrices_f16, to_npz       # noqa: E402
from wnn.metric import GlobalPearsonLoss, Moments        # noqa: E402
from wnn.models import build, featurize, mingru_scan     # noqa: E402


def _solution():
    spec = importlib.util.spec_from_file_location("wnn_solution", ROOT / "inference" / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ref_wp(t, p):
    t = np.clip(t.astype(np.float64), -2, 2); p = np.clip(p.astype(np.float64), -2, 2)
    w = np.abs(t); W = w.sum()
    mt, mp = (w * t).sum() / W, (w * p).sum() / W
    cov = (w * (t - mt) * (p - mp)).sum() / W
    return cov / np.sqrt((w * (t - mt) ** 2).sum() / W * (w * (p - mp) ** 2).sum() / W)


def test_moments_match_reference():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((5000, 2)).astype(np.float32) * 1.5
    p = (0.5 * y + rng.standard_normal((5000, 2))).astype(np.float32)
    m = Moments(); m.add(y, p, np.ones(5000, bool))
    ref = [_ref_wp(y[:, k], p[:, k]) for k in range(2)]
    assert np.allclose(m.per_target(), ref, atol=1e-10)


def test_loss_equals_metric():
    rng = np.random.default_rng(1)
    y = rng.standard_normal((4, 300, 2)).astype(np.float32)
    p = np.clip(0.3 * y + rng.standard_normal((4, 300, 2)), -2, 2).astype(np.float32)
    loss, corr = GlobalPearsonLoss(0.0, 0.0)(torch.tensor(p), torch.tensor(y), torch.ones(4, 300))
    m = Moments(); m.add(y.reshape(-1, 2), p.reshape(-1, 2), np.ones(1200, bool))
    assert abs(-float(loss) - m.score()) < 1e-5


@pytest.mark.parametrize("decay", [0.0, 0.9])
def test_micro_batch_gradient_is_exact(decay):
    """Two-pass micro-batching reproduces the full-batch gradient of the global Pearson loss."""
    torch.manual_seed(0)
    y = torch.randn(8, 50, 2); rw = torch.rand(8, 50); wy = y + 0.1 * torch.randn_like(y)
    prev = torch.randn(8, 50, 2)
    full = GlobalPearsonLoss(decay, 0.05)
    full(prev, y, rw, wy)                                # one earlier batch -> running sums
    p = torch.randn(8, 50, 2, requires_grad=True)
    full(p, y, rw, wy)[0].backward()
    micro = GlobalPearsonLoss(decay, 0.05)
    micro(prev, y, rw, wy)
    q = p.detach().clone().requires_grad_(True)
    with torch.no_grad():
        b = sum(micro.moments(q[s], y[s], rw[s], wy[s]) for s in (slice(0, 3), slice(3, 8)))
    tot, share = micro.total(b)
    for s in (slice(0, 3), slice(3, 8)):
        (micro.grad_coef(q[s], y[s], rw[s], wy[s], tot, share, b[:, 0].sum()) * q[s]).sum().backward()
    assert torch.allclose(p.grad, q.grad, rtol=1e-4, atol=1e-7)


def test_mingru_scan_matches_loop():
    torch.manual_seed(0)
    a, b = torch.randn(3, 700, 8) * 3, torch.randn(3, 700, 8) * 2
    h = mingru_scan(a, b, chunk=128)
    z = torch.sigmoid(a); g = torch.where(b >= 0, b + 0.5, torch.sigmoid(b))
    hs, cur = [], torch.zeros(3, 8)
    for t in range(700):
        cur = (1 - z[:, t]) * cur + z[:, t] * g[:, t]; hs.append(cur)
    assert torch.allclose(h, torch.stack(hs, 1), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("weights", ["f32", "f16"])
@pytest.mark.parametrize("kind", ["gru", "lrnn", "hybrid"])
def test_torch_vs_solution_step(kind, weights, tmp_path):
    cfg = load_config(None, [f"model.kind={kind}", "model.d_proj=16", "model.gru_hidden=16", "model.gru_layers=2",
                             "model.lrnn_layers=2", "model.lrnn_state=24", "model.lrnn_mlp=8",
                             "model.reset_every=50", "model.aux_horizons=[3]"])
    torch.manual_seed(0)
    net = build(cfg).eval()
    for p in net.parameters():                       # non-trivial weights everywhere (blocks start at zero)
        p.data.add_(torch.randn_like(p) * 0.2)
    if weights == "f16":
        round_matrices_f16(net)
    rng = np.random.default_rng(0)
    T = 200
    mean = rng.standard_normal(112).astype(np.float32)
    std = (rng.random(112) + 0.5).astype(np.float32)
    raw = (mean + std * rng.standard_normal((T, 112))).astype(np.float16).astype(np.float32)
    # torch: segments of reset_every rows, each from a zero state; dx uses the row before
    padded = torch.from_numpy(np.concatenate([raw[:1], raw]))[None]
    f = featurize(padded, torch.from_numpy(mean), torch.from_numpy(std), cfg.features.clip, True)
    with torch.no_grad():
        ref, _ = net(f.reshape(T // 50, 50, -1))
    ref = ref.reshape(T, 2).numpy()
    st = types.SimpleNamespace(mean=mean, std=std)
    Wt, meta = to_npz(net, cfg, st, weights)
    np.savez(tmp_path / "model.npz", **Wt)
    meta["backend"] = "numpy"
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    sol = _solution()
    sol.HERE = tmp_path
    m = sol.PredictionModel("numpy")
    out = np.stack([m.predict(types.SimpleNamespace(seq_ix=7, step_in_seq=t, need_prediction=True, state=raw[t]))
                    for t in range(T)])
    assert np.abs(out - ref).max() < 1e-4, np.abs(out - ref).max()
