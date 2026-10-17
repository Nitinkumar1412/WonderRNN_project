"""Check a submission folder the way the scorer runs it (no torch, no wnn package needed).

    python tools/check_submission.py <submission dir> <valid.parquet> <folder with utils.py> [n_seqs]

For each backend that loads (C kernel, numpy): replays n valid sequences through
PredictionModel.predict with the scorer's DataPoint, reports us/row, the projected
wall time for 39.4M rows, the masked WP, and the max difference between backends.
Run inside the scorer replica with 1 CPU: tools/docker_check.sh.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    sub, valid, starter = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    sys.path.insert(0, str(starter))
    from utils import FEATURE_COLUMNS, TARGET_COLUMNS, DataPoint, GlobalAccumulator
    spec = importlib.util.spec_from_file_location("solution", sub / "solution.py")
    sol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sol)

    pf = pq.ParquetFile(valid)
    seqs = []
    for rg in range(n):
        t = pf.read_row_group(rg)
        seqs.append((int(t.column("seq_ix")[0].as_py()),
                     np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32),
                     np.column_stack([t[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32),
                     t["need_prediction"].to_numpy().astype(bool),
                     t["is_scored"].to_numpy().astype(bool)))
    preds = {}
    for backend in ("c", "numpy"):
        t0 = time.perf_counter()
        try:
            m = sol.PredictionModel(backend)
        except Exception as e:                   # noqa: BLE001
            print(f"{backend:6s} unavailable: {e.__class__.__name__}: {e}")
            continue
        init = time.perf_counter() - t0
        acc, allp, rows, el = GlobalAccumulator(), [], 0, 0.0
        for seq, X, Y, need, scored in seqs:
            P = np.full((len(X), 2), np.nan, np.float32)
            t0 = time.perf_counter()
            for s in range(len(X)):
                v = m.predict(DataPoint(seq, s, bool(need[s]), X[s]))
                if v is not None:
                    P[s] = v
            el += time.perf_counter() - t0
            rows += len(X)
            acc.add(Y, P, scored & need)
            allp.append(P)
        us = el / rows * 1e6
        preds[backend] = np.concatenate(allp)
        print(f"{backend:6s} init {init:.1f}s  {us:6.1f} us/row  -> {us * 39.4e6 / 6e7:5.1f} min / 39.4M rows  "
              f"WP {acc.result()['weighted_pearson']:.5f}")
    best = "c" if "c" in preds else ("numpy" if "numpy" in preds else None)
    if best:                                     # the rules ask for identical predictions on repeated runs
        seq, X, _, need, _ = seqs[0]
        m = sol.PredictionModel(best)
        P = np.full((len(X), 2), np.nan, np.float32)
        for s in range(len(X)):
            v = m.predict(DataPoint(seq, s, bool(need[s]), X[s]))
            if v is not None:
                P[s] = v
        first = preds[best][:len(X)]
        print(f"determinism ({best}): second run bit-identical = {np.array_equal(P, first, equal_nan=True)}")
    if len(preds) == 2:
        ok = ~np.isnan(preds["c"][:, 0])
        print(f"max |c - numpy| = {np.abs(preds['c'][ok] - preds['numpy'][ok]).max():.2e}")


if __name__ == "__main__":
    main()
