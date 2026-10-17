"""export: a run -> submission folder + zip, then parity and latency checks  (CPU)

    python -m wnn export --run stu_f0 [export.name=sub_v3] [export.backend=numpy]

Writes work/submissions/<name>/{solution.py, kernel.c, model.npz, meta.json} and
work/submissions/<name>.zip (everything at the zip root). Then:
  parity  replays export.parity_seqs valid sequences from the parquet file through
          PredictionModel.predict (every backend that loads here) and compares with the
          torch model; reports the max difference and both WP scores
  bench   times PredictionModel.predict on export.bench_seqs sequences the way the
          scorer calls it and projects the wall time for export.test_rows rows
"""
from __future__ import annotations

import copy
import importlib.util
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from .config import CODE_DIR
from .data import FEATURE_COLUMNS, SEQ_LEN, TARGET_COLUMNS, WARMUP, Store
from .evaluate import predict
from .metric import global_wp
from .runs import load_run
from .util import Work, log, save_json

INFERENCE_DIR = CODE_DIR / "inference"


def round_matrices_f16(model) -> None:
    """Round every weight matrix to float16 in place, so the torch reference equals what ships."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2:
                p.copy_(p.half().float())


def to_npz(model, rcfg, st: Store, weights: str = "f32") -> tuple[dict, dict]:
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    mc = rcfg.model
    W = {"mean": st.mean, "std": st.std, "proj_w": sd["proj.weight"].T, "proj_b": sd["proj.bias"]}
    gl = int(mc.gru_layers) if mc.kind in ("gru", "hybrid") else 0
    nb = int(mc.lrnn_layers) if mc.kind in ("lrnn", "hybrid") else 0
    if gl > 4 or nb > 8:
        raise ValueError("kernel.c supports at most 4 GRU layers and 8 blocks")
    for l in range(gl):
        W[f"gru{l}_wi"] = sd[f"gru.weight_ih_l{l}"].T
        W[f"gru{l}_bi"] = sd[f"gru.bias_ih_l{l}"]
        W[f"gru{l}_wh"] = sd[f"gru.weight_hh_l{l}"].T
        W[f"gru{l}_bh"] = sd[f"gru.bias_hh_l{l}"]
    for k in range(nb):
        p, q = f"blocks.{k}.", f"blk{k}_"
        W[q + "n1"] = sd[p + "n1.weight"]
        W[q + "win"], W[q + "bin"] = sd[p + "inp.weight"].T, sd[p + "inp.bias"]
        W[q + "wout"], W[q + "bout"] = sd[p + "out.weight"].T, sd[p + "out.bias"]
        if int(mc.lrnn_mlp):
            W[q + "n2"] = sd[p + "n2.weight"]
            W[q + "w1"], W[q + "b1"] = sd[p + "fc1.weight"].T, sd[p + "fc1.bias"]
            W[q + "w2"], W[q + "b2"] = sd[p + "fc2.weight"].T, sd[p + "fc2.bias"]
    if nb:
        W["norm_f"] = sd["norm_f.weight"]
    W["head_w"], W["head_b"] = sd["head.weight"].T, sd["head.bias"]
    W = {k: np.ascontiguousarray(v, np.float32) for k, v in W.items()}
    if weights == "f16":
        mats = {"proj_w", "head_w"} | {k for k in W if k.endswith(("_wi", "_wh", "_win", "_wout", "_w1", "_w2"))}
        W = {k: (v.astype(np.float16) if k in mats else v) for k, v in W.items()}   # = solution.is_matrix
    meta = {"kind": mc.kind, "d_proj": int(mc.d_proj), "gru_hidden": int(mc.gru_hidden), "gru_layers": gl,
            "lrnn_layers": nb, "lrnn_state": int(mc.lrnn_state), "lrnn_mlp": int(mc.lrnn_mlp),
            "reset_every": int(mc.reset_every), "out_bound": float(mc.out_bound),
            "clip": float(rcfg.features.clip), "diff": bool(rcfg.features.diff), "dtype": rcfg.data.dtype,
            "weights": weights}
    return W, meta


def load_solution(sub_dir: Path):
    spec = importlib.util.spec_from_file_location(f"wnn_sub_{sub_dir.name}", sub_dir / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _datapoint_cls(cfg):
    sp = Path(cfg.paths.starterpack)
    if (sp / "utils.py").exists():
        sys.path.insert(0, str(sp))
        from utils import DataPoint            # the scorer's own class
        return DataPoint
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class DataPoint:                          # same fields as the starter pack's
        seq_ix: int
        step_in_seq: int
        need_prediction: bool
        state: np.ndarray
    return DataPoint


def _valid_sequences(cfg, st: Store, n: int):
    """First n valid sequences straight from the parquet (float32, exactly what the scorer feeds)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(Path(cfg.paths.data_dir) / "valid.parquet")
    by_rg = {int(src[1]): i for i, src in enumerate(st.index["source"]) if src[0] == "valid"}
    for rg in sorted(by_rg)[:n]:
        t = pf.read_row_group(rg, columns=["seq_ix", "is_scored"] + FEATURE_COLUMNS + list(TARGET_COLUMNS))
        X = np.stack([t.column(c).to_numpy() for c in FEATURE_COLUMNS], 1).astype(np.float32)
        Y = np.stack([t.column(c).to_numpy() for c in TARGET_COLUMNS], 1).astype(np.float32)
        yield by_rg[rg], int(t.column("seq_ix")[0].as_py()), X, Y, t.column("is_scored").to_numpy()


def replay(mod, backend: str, DataPoint, seq_ix: int, X: np.ndarray):
    m = mod.PredictionModel(backend)
    if backend != "auto" and m.backend != backend:
        raise RuntimeError(f"asked for {backend}, got {m.backend}")
    P = np.zeros((SEQ_LEN, 2), np.float32)
    t0 = time.perf_counter()
    for s in range(SEQ_LEN):
        v = m.predict(DataPoint(seq_ix, s, s >= WARMUP, X[s]))
        if v is not None:
            P[s] = v
    return P, (time.perf_counter() - t0) / SEQ_LEN * 1e6


def run(cfg, args) -> None:
    W = Work(cfg)
    if not args.run:
        raise SystemExit("export needs --run NAME")
    model, rcfg, ck = load_run(W.root, args.run, "cpu")
    if ck["objective"] != "target":
        raise SystemExit("only target runs can be exported")
    st = Store(W.arrays(""), rcfg.data.dtype, with_mask_p=False)
    name = cfg.export.name or args.run
    sub = W.root / "submissions" / name
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir(parents=True)
    weights = cfg.export.weights
    full = None
    if weights == "f16":
        full = copy.deepcopy(model)               # unrounded, to report what the rounding costs
        round_matrices_f16(model)
    Wt, meta = to_npz(model, rcfg, st, weights)
    meta.update({"backend": cfg.export.backend, "run": args.run})
    np.savez(sub / "model.npz", **Wt)
    save_json(meta, sub / "meta.json")
    shutil.copy(INFERENCE_DIR / "solution.py", sub / "solution.py")
    shutil.copy(INFERENCE_DIR / "kernel.c", sub / "kernel.c")
    zpath = W.root / "submissions" / f"{name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("solution.py", "kernel.c", "model.npz", "meta.json"):
            z.write(sub / f, f)
    mb = zpath.stat().st_size / 1e6
    log(f"wrote {zpath} ({mb:.2f} MB{' - OVER 20 MB!' if mb > 20 else ''})")

    mod = load_solution(sub)
    DataPoint = _datapoint_cls(cfg)
    report = {"zip_mb": mb, "run": args.run, "meta": meta}
    backends = ["numpy"]
    try:
        _, b = mod.make_net("c")
        backends.append("c")
    except Exception as e:                    # noqa: BLE001
        log(f"C backend not available here ({e.__class__.__name__}: {e}) - parity/bench on numpy only")
    torch.set_num_threads(1)
    for idx, seq_ix, X, Y, scored in _valid_sequences(cfg, st, int(cfg.export.parity_seqs)):
        _, ref = next(predict(model, rcfg, st, [idx], 1, "cpu"))
        need = np.arange(SEQ_LEN) >= WARMUP
        sel = need & scored.astype(bool)
        for b in backends:
            P, us = replay(mod, b, DataPoint, seq_ix, X)
            d = float(np.abs(P[need] - ref[need, :2]).max())
            r = {"max_abs_diff": d, "us_per_row": us,
                 "wp_solution": global_wp(Y, P, sel), "wp_torch": global_wp(Y, ref[:, :2], sel)}
            if full is not None:
                r["wp_torch_f32_weights"] = global_wp(Y, next(predict(full, rcfg, st, [idx], 1, "cpu"))[1][:, :2], sel)
            report[f"parity_{b}_seq{seq_ix}"] = r
            log(f"parity {b} seq {seq_ix}: max|diff| {d:.2e}  {us:.1f} us/row  "
                f"wp sol {r['wp_solution']:.4f} torch {r['wp_torch']:.4f}")
            if d > 2e-3:
                log(f"WARNING: {b} solution differs from torch by {d:.2e} - check before submitting")
    for b in backends:
        times = []
        for _, seq_ix, X, _, _ in _valid_sequences(cfg, st, int(cfg.export.bench_seqs)):
            times.append(replay(mod, b, DataPoint, seq_ix, X)[1])
        us = float(np.mean(times))
        mins = us * float(cfg.export.test_rows) / 6e7
        report[f"bench_{b}"] = {"us_per_row": us, "projected_min_here": mins}
        log(f"bench {b}: {us:.1f} us/row -> {mins:.1f} min for {cfg.export.test_rows/1e6:.1f}M rows on this CPU "
            f"(the scorer may be slower: calibrate with a real submission)")
    save_json(report, W.root / "submissions" / f"{name}_report.json")
