"""Training batches: random reset-segments of random sequences, read from the memmaps.

A segment is rows [s, s+L) of one sequence with s a multiple of L (L = reset_every,
or the whole 20k sequence when reset_every = 0): exactly what solution.py runs
between two state resets. Each item also carries the row before the segment
(for dx), targets extended by the largest aux horizon, the row weights and,
when distilling, the teacher's soft targets.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import SEQ_LEN, UNKNOWN, WARMUP, Store


def row_weights(scored: np.ndarray, mask_p: np.ndarray | None, is_valid: bool, steps: np.ndarray,
                source: str, floor: float) -> np.ndarray:
    """Per-row loss weight (before |y|): warm-up rows 0, else floor + (1-floor) * P(scored)."""
    need = (steps >= WARMUP).astype(np.float32)
    if source == "none":
        m = np.ones(len(steps), np.float32)
    elif is_valid and source in ("auto", "true_only"):
        m = (scored == 1).astype(np.float32)
    elif source == "true_only":
        return np.zeros(len(steps), np.float32)          # train sequences have no true mask
    else:
        m = mask_p.astype(np.float32) if mask_p is not None else np.ones(len(steps), np.float32)
    return need * (floor + (1.0 - floor) * m)


class SegmentBatches(torch.utils.data.IterableDataset):
    def __init__(self, arrays_dir: str, dtype: str, seq_ids: np.ndarray, seg_len: int, batch: int,
                 seed: int, epoch: int, objective: str, mask_source: str, mask_floor: float,
                 horizons: list[int], teacher_path: str = "", max_batches: int = 0) -> None:
        super().__init__()
        self.args = (arrays_dir, dtype)
        self.seq_ids, self.L, self.batch = np.asarray(seq_ids), seg_len, batch
        self.seed, self.epoch, self.objective = seed, epoch, objective
        self.mask_source, self.mask_floor = mask_source, mask_floor
        self.hmax = max(horizons) if horizons else 0
        self.teacher_path = teacher_path
        self.max_batches = max_batches

    def plan(self) -> list[np.ndarray]:
        starts = np.arange(0, SEQ_LEN, self.L)
        items = np.stack(np.meshgrid(self.seq_ids, starts, indexing="ij"), -1).reshape(-1, 2)
        rng = np.random.default_rng(self.seed * 1000 + self.epoch)
        items = items[rng.permutation(len(items))]
        out = [items[i:i + self.batch] for i in range(0, len(items) - self.batch + 1, self.batch)]
        if not out:                       # fewer items than one batch (smoke runs)
            out = [items]
        return out[: self.max_batches] if self.max_batches else out

    def __len__(self) -> int:
        return len(self.plan())

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        st = Store(*self.args)
        teacher = np.load(self.teacher_path, mmap_mode="r") if self.teacher_path else None
        for b in self.plan()[wid::nw]:
            yield self.load(st, teacher, b)

    def load(self, st: Store, teacher, items: np.ndarray) -> dict:
        L, H = self.L, self.hmax
        n = len(items)
        raw = np.empty((n, L + 1, st.X.shape[-1]), dtype=st.X.dtype)
        y = np.zeros((n, L + H, 2), np.float32)
        yok = np.zeros((n, L + H), np.float32)
        w = np.empty((n, L), np.float32)
        tgt_scored = np.zeros((n, L), np.float32)
        soft = np.empty((n, L, 2), np.float32) if teacher is not None else None
        steps = np.arange(L)
        for k, (i, s) in enumerate(items):
            i, s = int(i), int(s)
            if s == 0:
                raw[k, 1:] = st.X[i, :L]
                raw[k, 0] = raw[k, 1]                # dx = 0 at the first row of a sequence
            else:
                raw[k] = st.X[i, s - 1:s + L]
            e = min(s + L + H, SEQ_LEN)
            y[k, : e - s] = st.Y[i, s:e]
            yok[k, : e - s] = 1.0
            sc = np.asarray(st.scored[i, s:s + L])
            mp = np.asarray(st.mask_p[i, s:s + L]) if st.mask_p is not None else None
            w[k] = row_weights(sc, mp, bool(st.is_valid[i]), s + steps, self.mask_source, self.mask_floor)
            if self.objective == "mask":
                tgt_scored[k] = (sc == 1)
                w[k] = ((s + steps) >= WARMUP).astype(np.float32)
            if soft is not None:
                soft[k] = teacher[i, s:s + L]
        out = {"raw": torch.from_numpy(raw), "y": torch.from_numpy(y), "yok": torch.from_numpy(yok),
               "w": torch.from_numpy(w), "scored": torch.from_numpy(tgt_scored),
               "start": torch.from_numpy(items[:, 1].astype(np.int64))}
        if soft is not None:
            out["soft"] = torch.from_numpy(soft)
        return out
