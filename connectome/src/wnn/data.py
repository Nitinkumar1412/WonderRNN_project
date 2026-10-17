"""Constants and the on-disk array store shared by every step.

Store (work/arrays), one row per sequence, train sequences first, then valid:
    X.f16.npy      [N, T, 112]  raw features (float16 by default, see data.dtype)
    Y.f32.npy      [N, T, 2]    targets t0, t1
    scored.u8.npy  [N, T]       is_scored: 0/1 on valid sequences, 255 = unknown (train)
    mask_p.f16.npy [N, T]       P(is_scored) from the mask model (label-mask step), optional
    stats.npz      mean, std    feature standardisation (from the stored, rounded values)
    index.json     seq_ix, is_valid, fold, source per sequence
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .util import load_json

SEQ_LEN = 20_000
WARMUP = 99
N_FEATURES = 112
FEATURE_COLUMNS = [f"i{i}_{g}{j}" for i in range(2)
                   for g, n in [("p", 22), ("v", 22), ("dp", 4), ("dv", 4)] for j in range(n)]
FEATURE_COLUMNS += [f"a{i}" for i in range(8)]
TARGET_COLUMNS = ("t0", "t1")
METRIC_CLIP = 2.0
UNKNOWN = 255


def x_name(dtype: str) -> str:
    return "X.f16.npy" if dtype == "float16" else "X.f32.npy"


class Store:
    """Read-only view of the prepared arrays (memory-mapped; cheap to open in every worker)."""

    def __init__(self, arrays_dir: str | Path, dtype: str = "float16", with_mask_p: bool = True) -> None:
        d = Path(arrays_dir)
        self.dir = d
        self.index = load_json(d / "index.json")
        self.X = np.load(d / x_name(dtype), mmap_mode="r")
        self.Y = np.load(d / "Y.f32.npy", mmap_mode="r")
        self.scored = np.load(d / "scored.u8.npy", mmap_mode="r")
        mp = d / "mask_p.f16.npy"
        self.mask_p = np.load(mp, mmap_mode="r") if (with_mask_p and mp.exists()) else None
        st = np.load(d / "stats.npz")
        self.mean = st["mean"].astype(np.float32)
        self.std = st["std"].astype(np.float32)
        self.is_valid = np.asarray(self.index["is_valid"], dtype=bool)
        self.fold = np.asarray(self.index["fold"], dtype=np.int64)
        self.seq_ix = np.asarray(self.index["seq_ix"], dtype=np.int64)

    @property
    def n(self) -> int:
        return len(self.seq_ix)

    def split(self, fold: int | None, objective: str = "target") -> tuple[np.ndarray, np.ndarray]:
        """(train_ids, heldout_ids). fold None/-1 -> all sequences train, no held-out.
        The mask objective only has labels on valid sequences."""
        ids = np.arange(self.n)
        if objective == "mask":
            ids = ids[self.is_valid]
        if fold is None or fold < 0:
            return ids, ids[:0]
        return ids[self.fold[ids] != fold], ids[self.fold[ids] == fold]
