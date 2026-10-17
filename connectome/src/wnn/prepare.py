"""prepare: competition parquet -> memory-mapped arrays + feature stats + folds  (CPU)

Each parquet row group is one complete sequence, so every worker process reads
one row group and writes its slice of the preallocated .npy files in place.
Train sequences come first, then valid. Re-running rebuilds everything.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import math
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import numpy as np

from .data import (FEATURE_COLUMNS, N_FEATURES, SEQ_LEN, TARGET_COLUMNS, UNKNOWN, WARMUP,
                   x_name)
from .util import Work, log, n_threads, save_json, timed


def stable_fold(seq_ix: int, n_folds: int) -> int:
    return int(hashlib.md5(str(int(seq_ix)).encode()).hexdigest(), 16) % n_folds


def _write_one(task: tuple) -> tuple[int, int]:
    path, rg, out, arrays_dir, xname, has_mask = task
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = ["seq_ix", "step_in_seq", "need_prediction"] + (["is_scored"] if has_mask else [])
    t = pf.read_row_group(rg, columns=cols + FEATURE_COLUMNS + list(TARGET_COLUMNS), use_threads=False)
    seq = t.column("seq_ix").to_numpy()
    step = t.column("step_in_seq").to_numpy()
    if len(seq) != SEQ_LEN or (seq != seq[0]).any() or not np.array_equal(step, np.arange(SEQ_LEN)):
        raise ValueError(f"{path} row group {rg}: not one ordered {SEQ_LEN}-row sequence")
    need = t.column("need_prediction").to_numpy()
    if need[:WARMUP].any() or not need[WARMUP:].all():
        raise ValueError(f"{path} row group {rg}: unexpected need_prediction layout")
    d = Path(arrays_dir)
    X = np.load(d / xname, mmap_mode="r+")
    Y = np.load(d / "Y.f32.npy", mmap_mode="r+")
    S = np.load(d / "scored.u8.npy", mmap_mode="r+")
    X[out] = np.stack([t.column(c).to_numpy() for c in FEATURE_COLUMNS], axis=1).astype(X.dtype)
    Y[out] = np.stack([t.column(c).to_numpy() for c in TARGET_COLUMNS], axis=1).astype(np.float32)
    S[out] = t.column("is_scored").to_numpy().astype(np.uint8) if has_mask else UNKNOWN
    X.flush(); Y.flush(); S.flush()
    return out, int(seq[0])


def run(cfg, args) -> None:
    import pyarrow.parquet as pq
    W = Work(cfg)
    data_dir = Path(cfg.paths.data_dir)
    train_p, valid_p = data_dir / "train.parquet", data_dir / "valid.parquet"
    for p in (train_p, valid_p):
        if not p.exists():
            raise FileNotFoundError(f"missing {p} (set paths.data_dir / WNN_DATA_DIR)")
    n_tr, n_va = pq.ParquetFile(train_p).num_row_groups, pq.ParquetFile(valid_p).num_row_groups
    if cfg.data.max_seqs and cfg.data.max_seqs > 0:
        n_tr = min(n_tr, int(cfg.data.max_seqs))
        n_va = min(n_va, max(1, math.ceil(cfg.data.max_seqs / 6)))
    n = n_tr + n_va
    xname = x_name(cfg.data.dtype)
    dtype = np.float16 if cfg.data.dtype == "float16" else np.float32
    arrays = W.arrays("")
    gb = n * SEQ_LEN * N_FEATURES * np.dtype(dtype).itemsize / 1e9
    log(f"{n_tr} train + {n_va} valid sequences -> {arrays}  (X {gb:.1f} GB, {cfg.data.dtype})")

    shapes = {xname: (dtype, (n, SEQ_LEN, N_FEATURES)), "Y.f32.npy": (np.float32, (n, SEQ_LEN, 2)),
              "scored.u8.npy": (np.uint8, (n, SEQ_LEN))}
    done_path = arrays / "prepare_done.npy"           # seq_ix written into each slot, -1 = not yet
    resume = bool(cfg.data.get("resume", True)) and done_path.exists() and all(
        (arrays / f).exists() and np.load(arrays / f, mmap_mode="r").shape == s for f, (_, s) in shapes.items())
    if resume:
        done = np.load(done_path)
        resume = done.shape == (n,)
    if not resume:
        done = np.full(n, -1, np.int64)
    # the .npy files are sparse until written: check the disk can hold what is still missing
    need = sum(np.dtype(dt).itemsize * int(np.prod(s)) for dt, s in shapes.values())
    have = sum((arrays / f).stat().st_blocks * 512 for f in shapes if (arrays / f).exists()) \
        if hasattr(os.stat_result, "st_blocks") else 0
    free = shutil.disk_usage(arrays).free
    if need - (have if resume else 0) > free * 0.98:
        raise SystemExit(f"not enough disk at {arrays}: arrays need {need/1e9:.1f} GB, "
                         f"{free/1e9:.1f} GB free{' (+ %.1f GB already written)' % (have/1e9) if resume else ''}. "
                         f"Point WNN_WORK_DIR at bigger storage (e.g. /scratch/$USER/wnn_work) and re-run.")
    if not resume:
        for f, (dt, s) in shapes.items():
            np.lib.format.open_memmap(arrays / f, mode="w+", dtype=dt, shape=s)
        np.save(done_path, done)
        stale = arrays / "mask_p.f16.npy"
        if stale.exists():
            stale.unlink()              # belongs to the previous store
    else:
        log(f"resuming: {int((done >= 0).sum())}/{n} sequences already written")

    tasks = [(str(train_p), i, i, str(arrays), xname, False) for i in range(n_tr)]
    tasks += [(str(valid_p), i, n_tr + i, str(arrays), xname, True) for i in range(n_va)]
    tasks = [t for t in tasks if done[t[2]] < 0]
    workers = max(1, min(n_threads(cfg.data.workers), 32))

    def record(k: int, out: int, s: int) -> None:
        done[out] = s
        if k % 200 == 0 or k == len(tasks):
            np.save(arrays / "prepare_done.tmp.npy", done)
            (arrays / "prepare_done.tmp.npy").replace(done_path)
        if k % 500 == 0 or k == len(tasks):
            log(f"  {int((done >= 0).sum())}/{n}")

    with timed(f"writing {len(tasks)} sequences with {workers} processes"):
        if workers == 1:
            for k, t in enumerate(tasks, 1):
                record(k, *_write_one(t))
        else:
            # ProcessPoolExecutor raises BrokenProcessPool when a worker dies (mp.Pool hangs forever)
            ctx = mp.get_context("spawn")
            try:
                with cf.ProcessPoolExecutor(workers, mp_context=ctx) as ex:
                    futs = [ex.submit(_write_one, t) for t in tasks]
                    for k, fut in enumerate(cf.as_completed(futs), 1):
                        record(k, *fut.result())
            except cf.process.BrokenProcessPool as e:
                np.save(done_path, done)
                raise SystemExit(
                    f"a prepare worker died ({e}). Usual causes: disk full / quota (writing a memmap on a full "
                    f"disk kills the process with SIGBUS) or the job's memory limit. Check `df -h`, `quota -s`, "
                    f"`sacct -j $SLURM_JOB_ID -o JobID,State,MaxRSS`; re-running resumes from "
                    f"{int((done >= 0).sum())}/{n} written sequences.") from e
    seq_ix = done.tolist()
    if min(seq_ix) < 0:
        raise SystemExit(f"{int((done < 0).sum())} sequences were not written - re-run prepare")
    if len(set(seq_ix)) != n:
        raise ValueError("duplicate seq_ix across train/valid")

    X = np.load(arrays / xname, mmap_mode="r")
    rng = np.random.default_rng(cfg.seed)
    pick = np.sort(rng.choice(n, size=min(n, int(cfg.data.stats_seqs)), replace=False))
    s1 = np.zeros(N_FEATURES); s2 = np.zeros(N_FEATURES); cnt = 0
    for i in pick:
        x = X[i].astype(np.float64)
        s1 += x.sum(0); s2 += (x * x).sum(0); cnt += len(x)
    mean = s1 / cnt
    std = np.sqrt(np.maximum(s2 / cnt - mean * mean, 0.0))
    std = np.where(std < 1e-6, 1.0, std)
    np.savez(arrays / "stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    is_valid = [False] * n_tr + [True] * n_va
    folds = [stable_fold(s, cfg.data.n_folds) for s in seq_ix]
    source = [["train", i] for i in range(n_tr)] + [["valid", i] for i in range(n_va)]
    save_json({"seq_ix": seq_ix, "is_valid": is_valid, "fold": folds, "source": source,
               "n_folds": cfg.data.n_folds, "dtype": cfg.data.dtype}, arrays / "index.json")
    counts = np.bincount(folds, minlength=cfg.data.n_folds).tolist()
    log(f"stats from {len(pick)} seqs: |mean| max {np.abs(mean).max():.3f}, std range "
        f"[{std.min():.3f}, {std.max():.3f}]; fold sizes {counts}")
