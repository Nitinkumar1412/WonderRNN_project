"""Global weighted Pearson: the exact numpy scorer and the torch training loss.

Metric (METRIC.md): per target k over all selected rows of the dataset,
    y = clip(t, -2, 2), p = clip(pred, -2, 2), w = |y|,  rho_k = weighted Pearson(y, p; w)
    score = (rho_0 + rho_1) / 2
"""
from __future__ import annotations

import numpy as np

from .data import METRIC_CLIP


class Moments:
    """Weighted moments merged across sequences (float64), like utils.GlobalAccumulator."""

    def __init__(self) -> None:
        self.s = np.zeros((2, 6))           # per target: W, Sy, Sp, Syy, Spp, Syp

    def add(self, y: np.ndarray, p: np.ndarray, sel: np.ndarray) -> None:
        """y, p: [..., 2]; sel: [...] bool rows to score."""
        y = np.clip(y[sel].astype(np.float64), -METRIC_CLIP, METRIC_CLIP)
        p = np.clip(p[sel].astype(np.float32).astype(np.float64), -METRIC_CLIP, METRIC_CLIP)
        w = np.abs(y)
        self.s += np.stack([w.sum(0), (w * y).sum(0), (w * p).sum(0), (w * y * y).sum(0),
                            (w * p * p).sum(0), (w * y * p).sum(0)], axis=1)

    def per_target(self) -> list[float]:
        out = []
        for W, Sy, Sp, Syy, Spp, Syp in self.s:
            if W < 1e-8:
                out.append(0.0); continue
            my, mp = Sy / W, Sp / W
            cov = Syp / W - my * mp
            vy, vp = Syy / W - my * my, Spp / W - mp * mp
            if vy <= 1e-16 or vp <= 1e-16:
                out.append(0.0); continue
            out.append(float(np.clip(cov / np.sqrt(vy * vp), -1.0, 1.0)))
        return out

    def score(self) -> float:
        return float(np.mean(self.per_target()))


def global_wp(y: np.ndarray, p: np.ndarray, sel: np.ndarray) -> float:
    m = Moments()
    m.add(y, p, sel)
    return m.score()


# ---------------------------------------------------------------- torch losses
def _torch():
    import torch
    return torch


class GlobalPearsonLoss:
    """-(mean over targets of the weighted Pearson) on a whole batch, rows pooled.

    w = |clip(y)| * row_w. With moment_ema > 0 the batch moments are added to an
    exponential running sum of earlier batches (detached), so the correlation is
    closer to the dataset-global one the metric uses; the loss is divided by the
    batch's share of the total weight to keep gradient magnitudes comparable.
    """

    def __init__(self, moment_ema: float = 0.0, mse_weight: float = 0.0) -> None:
        self.decay = float(moment_ema)
        self.mse_weight = float(mse_weight)
        self.run = None

    @staticmethod
    def _prep(p, y, row_w, wy):
        yc = y.float().clamp(-METRIC_CLIP, METRIC_CLIP)
        wsrc = yc if wy is None else wy.float().clamp(-METRIC_CLIP, METRIC_CLIP)
        w = wsrc.abs() * row_w.unsqueeze(-1)                  # [B, T, 2]
        return w.reshape(-1, 2).double(), yc.reshape(-1, 2).double(), p.float().reshape(-1, 2).double()

    def moments(self, p, y, row_w, wy=None):
        """Weighted sums [2 targets, 6] (W, Sy, Sp, Syy, Spp, Syp) of one (micro-)batch."""
        torch = _torch()
        w, yc, p = self._prep(p, y, row_w, wy)
        return torch.stack([w.sum(0), (w * yc).sum(0), (w * p).sum(0), (w * yc * yc).sum(0),
                            (w * p * p).sum(0), (w * yc * p).sum(0)], dim=1)

    def total(self, b):
        """Combine batch sums with the running sums (and advance them): -> (tot, share)."""
        torch = _torch()
        if self.decay > 0 and self.run is not None:
            tot = self.decay * self.run.to(b.device) + b
            share = (b[:, 0] / tot[:, 0].clamp_min(1e-8)).detach()
        else:
            tot, share = b, torch.ones(2, dtype=b.dtype, device=b.device)
        if self.decay > 0:
            self.run = tot.detach()
        return tot, share

    @staticmethod
    def _stats(tot):
        torch = _torch()
        W = tot[:, 0].clamp_min(1e-8)
        my, mp = tot[:, 1] / W, tot[:, 2] / W
        cov = tot[:, 5] / W - my * mp
        vy = (tot[:, 3] / W - my * my).clamp_min(1e-12)
        vp = (tot[:, 4] / W - mp * mp).clamp_min(1e-12)
        return W, my, mp, cov, vy, vp, cov / torch.sqrt(vy * vp)

    def __call__(self, p, y, row_w, wy=None):
        """p, y: [B, T, 2]; row_w: [B, T]; wy: the targets the weights come from (default y;
        distillation passes the true targets while y is the blended one)."""
        w, yc, pd = self._prep(p, y, row_w, wy)
        tot, share = self.total(self.moments(p, y, row_w, wy))
        corr = self._stats(tot)[-1]
        loss = -(corr / share).mean()
        if self.mse_weight > 0:
            loss = loss + self.mse_weight * (w * (pd - yc) ** 2).sum() / w.sum().clamp_min(1e-8)
        return loss.float(), corr.detach().float()

    def grad_coef(self, p, y, row_w, wy, tot, share, w_batch):
        """dLoss/dp for one micro-batch, given the sums `tot` of the WHOLE batch (from a no-grad
        pass) and the whole batch's total weight `w_batch`: backpropagating (coef * p).sum()
        over the micro-batches gives exactly the full-batch gradient of __call__."""
        torch = _torch()
        w, yc, pd = self._prep(p.detach(), y, row_w, wy)
        W, my, mp, cov, vy, vp, _ = self._stats(tot)
        dcorr = w / W * ((yc - my) / torch.sqrt(vy * vp) - cov * (pd - mp) / (torch.sqrt(vy) * vp ** 1.5))
        g = -dcorr / share / 2.0
        if self.mse_weight > 0:
            g = g + self.mse_weight * 2.0 * w * (pd - yc) / w_batch.clamp_min(1e-8)
        return g.reshape(p.shape).float()

    @classmethod
    def corr_of(cls, tot):
        return cls._stats(tot)[-1].detach().float()


def weighted_mse(p, y, w):
    return (w * (p.float() - y.float()) ** 2).sum() / w.sum().clamp_min(1e-8)
