"""Models: one network family, three shapes, all O(1) per step at inference.

    features  f_t = [x_t, x_t - x_{t-1}]          x = clip((raw - mean) / std, +-clip)
    proj      u = gelu_tanh(W_p f + b_p)                          (d_proj)
    gru       u = GRU^L(u)                                        kind gru | hybrid
    blocks    u = LBlock^K(u); u = rmsnorm(u)                     kind lrnn | hybrid
    head      y = B * tanh(W_h u / B)                             2 outputs, bounded by out_bound

LBlock (input-gated linear recurrence, minGRU-style):
    v = rmsnorm(u);  [a | b] = W_in v
    z = sigmoid(a);  c = g(b) (> 0);  h_t = (1 - z) * h_{t-1} + z * c          (elementwise)
    u = u + W_out h;  u = u + W_2 gelu_tanh(W_1 rmsnorm(u))
Training runs the recurrence as a chunked log-space parallel scan (fp32);
inference (solution.py / kernel.c) runs the same equations step by step.

Every segment starts from a zero state: with reset_every = R the training
segments are exactly the inference segments [kR, (k+1)R).
"""
from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .data import N_FEATURES

RMS_EPS = 1e-6


def gelu(x):
    return F.gelu(x, approximate="tanh")


class RMSNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        return (y * self.weight.float()).to(x.dtype)


def log_g(b):
    """log of g(b) = b + 0.5 (b >= 0) | sigmoid(b) (b < 0); g > 0 keeps the scan in log space."""
    return torch.where(b >= 0, torch.log(F.relu(b) + 0.5), -F.softplus(-b))


def mingru_scan(a, b, chunk: int = 256):
    """h_t = (1 - sigmoid(a_t)) h_{t-1} + sigmoid(a_t) g(b_t), h_{-1} = 0.  a, b: [B, T, S] -> h fp32."""
    a = a.float(); b = b.float()
    log_coef = -F.softplus(a)                       # log(1 - z)
    log_val = -F.softplus(-a) + log_g(b)            # log(z * g(b))
    T = a.shape[1]
    out = []
    log_h = torch.full_like(a[:, :1], float("-inf"))
    for s in range(0, T, chunk):
        lc, lv = log_coef[:, s:s + chunk], log_val[:, s:s + chunk]
        a_star = torch.cumsum(lc, dim=1)
        lh = a_star + torch.logaddexp(log_h, torch.logcumsumexp(lv - a_star, dim=1))
        out.append(lh)
        log_h = lh[:, -1:]
    return torch.exp(torch.cat(out, dim=1))


class LBlock(nn.Module):
    def __init__(self, d: int, state: int, mlp: int) -> None:
        super().__init__()
        self.n1 = RMSNorm(d)
        self.inp = nn.Linear(d, 2 * state)
        self.out = nn.Linear(state, d)
        self.mlp = mlp
        if mlp:
            self.n2 = RMSNorm(d)
            self.fc1 = nn.Linear(d, mlp)
            self.fc2 = nn.Linear(mlp, d)
        nn.init.zeros_(self.out.weight)             # blocks start as identity
        nn.init.zeros_(self.out.bias)
        if mlp:
            nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        a, b = self.inp(self.n1(x)).chunk(2, dim=-1)
        h = mingru_scan(a, b)
        x = x + self.out(h.to(x.dtype))
        if self.mlp:
            x = x + self.fc2(gelu(self.fc1(self.n2(x))))
        return x


def n_inputs(cfg) -> int:
    return N_FEATURES * (2 if cfg.features.diff else 1)


class Net(nn.Module):
    def __init__(self, mc, n_in: int, n_out: int = 2, bound: float = 0.0, n_aux: int = 0) -> None:
        super().__init__()
        self.kind = mc.kind
        bidir = bool(mc.get("bidirectional", False))
        if bidir and self.kind != "gru":
            raise ValueError("model.bidirectional needs model.kind=gru")
        if self.kind not in ("gru", "lrnn", "hybrid"):
            raise ValueError(f"model.kind must be gru | lrnn | hybrid, got {self.kind}")
        d = int(mc.d_proj)
        self.proj = nn.Linear(n_in, d)
        width = d
        if self.kind in ("gru", "hybrid"):
            if self.kind == "hybrid" and int(mc.gru_hidden) != d:
                raise ValueError("hybrid needs model.gru_hidden == model.d_proj")
            self.gru = nn.GRU(d, int(mc.gru_hidden), int(mc.gru_layers), batch_first=True, bidirectional=bidir)
            width = int(mc.gru_hidden) * (2 if bidir else 1)
        if self.kind in ("lrnn", "hybrid"):
            self.blocks = nn.ModuleList(LBlock(d, int(mc.lrnn_state), int(mc.lrnn_mlp))
                                        for _ in range(int(mc.lrnn_layers)))
            self.norm_f = RMSNorm(d)
        self.head = nn.Linear(width, n_out)
        self.aux = nn.Linear(width, n_aux) if n_aux else None
        self.bound = float(bound)
        self.ckpt = False            # activation checkpointing (train.checkpoint / set on CUDA OOM)

    def _gru(self, u):
        return self.gru(u)[0]

    def forward(self, f):
        ck = self.ckpt and torch.is_grad_enabled()
        u = gelu(self.proj(f))
        if self.kind in ("gru", "hybrid"):
            u = checkpoint(self._gru, u, use_reentrant=False) if ck else self._gru(u)
        if self.kind in ("lrnn", "hybrid"):
            for blk in self.blocks:
                u = checkpoint(blk, u, use_reentrant=False) if ck else blk(u)
            u = self.norm_f(u)
        y = self.head(u).float()
        if self.bound > 0:
            y = self.bound * torch.tanh(y / self.bound)
        aux = self.aux(u).float() if self.aux is not None else None
        return y, aux


def featurize(raw, mean, std, clip: float, diff: bool):
    """raw [B, L+1, 112] (row 0 = the row before the segment) -> features [B, L, 112 or 224]."""
    x = ((raw.float() - mean) / std).clamp(-clip, clip)
    if not diff:
        return x[:, 1:]
    return torch.cat([x[:, 1:], x[:, 1:] - x[:, :-1]], dim=-1)


def build(cfg, objective: str = "target") -> Net:
    mc = cfg.model
    n_in = n_inputs(cfg)
    if bool(mc.get("bidirectional", False)) and objective != "mask":
        raise ValueError("model.bidirectional reads future rows: only allowed for objective=mask (never shipped)")
    if objective == "target":
        return Net(mc, n_in, 2, float(mc.out_bound), 2 * len(mc.aux_horizons or []))
    if objective == "mask":
        return Net(mc, n_in, 1, 0.0, 0)
    if objective == "pretrain":
        return Net(mc, n_in, N_FEATURES, 0.0, 0)
    raise ValueError(f"unknown objective {objective}")


def load_backbone(model: Net, state: dict) -> list[str]:
    """Load every matching weight except the heads; returns the loaded keys."""
    own = model.state_dict()
    keep = {k: v for k, v in state.items()
            if not k.startswith(("head.", "aux.")) and k in own and own[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    return sorted(keep)


def count_step_macs(cfg) -> int:
    """Multiply-accumulates per inference step (for the latency budget)."""
    mc = cfg.model
    n_in, d = n_inputs(cfg), int(mc.d_proj)
    macs = n_in * d + 2 * (d if mc.kind == "lrnn" else int(mc.gru_hidden))
    if mc.kind in ("gru", "hybrid"):
        h, inp = int(mc.gru_hidden), d
        for _ in range(int(mc.gru_layers)):
            macs += 3 * h * (inp + h); inp = h
    if mc.kind in ("lrnn", "hybrid"):
        s, m = int(mc.lrnn_state), int(mc.lrnn_mlp)
        macs += int(mc.lrnn_layers) * (d * 2 * s + s * d + 2 * d * m)
    return macs


def segment_len(cfg) -> int:
    from .data import SEQ_LEN
    r = int(cfg.model.reset_every)
    if r <= 0:
        return SEQ_LEN
    if SEQ_LEN % r:
        raise ValueError(f"model.reset_every={r} must divide {SEQ_LEN}")
    return r


__all__ = ["Net", "build", "featurize", "load_backbone", "count_step_macs", "segment_len",
           "mingru_scan", "log_g", "RMS_EPS"]
