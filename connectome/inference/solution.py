"""Alfa Connectome submission: one stateful recurrent network, one row per call.

Files next to this one: model.npz (weights, input-major), meta.json (shape + settings),
kernel.c (C step, compiled here at start-up). Backend "auto" compiles kernel.c with the
system gcc and checks it against the numpy step on random rows; any problem falls back
to numpy. Both run the same equations as training (src/wnn/models.py):

    x = clip((round_f16(raw) - mean) / std, +-clip);  f = [x, x - x_prev]
    u = gelu(proj f) -> GRU layers and/or linear-recurrence blocks -> head -> B*tanh(./B)

The recurrent state is zeroed when seq_ix changes and every `reset_every` rows.

Ensemble: meta.json "members": N runs N networks (member 0 = model.npz / meta.json, member i =
model_i.npz / meta_i.json) and returns the mean of their outputs. kernel.c is compiled once.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import ctypes
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KINDS = {"gru": 0, "lrnn": 1, "hybrid": 2}
MAX_GRU, MAX_BLK = 4, 8
BLK_KEYS = ("n1", "win", "bin", "wout", "bout", "n2", "w1", "b1", "w2", "b2")
MATRIX_SUFFIXES = ("_wi", "_wh", "_win", "_wout", "_w1", "_w2")


def is_matrix(name: str) -> bool:
    """Weight matrices (stored float16 when meta weights = f16); vectors stay float32."""
    return name in ("proj_w", "head_w") or name.endswith(MATRIX_SUFFIXES)


def _sigm(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gelu(x):
    return np.float32(0.5) * x * (np.float32(1.0) + np.tanh(
        np.float32(0.7978845608028654) * (x + np.float32(0.044715) * x * x * x)))


def _rms(x, g):
    return x * (np.float32(1.0) / np.sqrt(np.dot(x, x) / np.float32(len(x)) + np.float32(1e-6))) * g


class NumpyNet:
    """Reference step in numpy (always available)."""

    def __init__(self, Wt: dict, meta: dict) -> None:
        self.W, self.m = Wt, meta
        self.kind = meta["kind"]
        self.gl = meta["gru_layers"] if self.kind in ("gru", "hybrid") else 0
        self.nb = meta["lrnn_layers"] if self.kind in ("lrnn", "hybrid") else 0
        self.gh, self.S = meta["gru_hidden"], meta["lrnn_state"]
        self.clip, self.bound, self.diff = np.float32(meta["clip"]), np.float32(meta["out_bound"]), meta["diff"]
        self.mean, self.inv = Wt["mean"], (np.float32(1.0) / Wt["std"]).astype(np.float32)
        self.h = [np.zeros(self.gh, np.float32) for _ in range(self.gl)]
        self.hb = [np.zeros(self.S, np.float32) for _ in range(self.nb)]
        self.prev = np.zeros(len(self.mean), np.float32)

    def reset(self) -> None:
        for h in self.h + self.hb:
            h[:] = 0.0

    def step(self, raw: np.ndarray, first: bool) -> np.ndarray:
        W = self.W
        x = np.clip((raw - self.mean) / W["std"], -self.clip, self.clip).astype(np.float32)
        if self.diff:
            f = np.concatenate([x, np.zeros_like(x) if first else x - self.prev])
        else:
            f = x
        self.prev = x
        u = _gelu(f @ W["proj_w"] + W["proj_b"])
        for l in range(self.gl):
            gh, h = self.gh, self.h[l]
            a = u @ W[f"gru{l}_wi"] + W[f"gru{l}_bi"]
            b = h @ W[f"gru{l}_wh"] + W[f"gru{l}_bh"]
            r = _sigm(a[:gh] + b[:gh]); z = _sigm(a[gh:2 * gh] + b[gh:2 * gh])
            n = np.tanh(a[2 * gh:] + r * b[2 * gh:])
            h = ((np.float32(1.0) - z) * n + z * h).astype(np.float32)
            self.h[l] = h
            u = h
        if self.nb:
            for k in range(self.nb):
                p, S = f"blk{k}_", self.S
                a = _rms(u, W[p + "n1"]) @ W[p + "win"] + W[p + "bin"]
                z = _sigm(a[:S]); bb = a[S:]
                g = np.where(bb >= 0, bb + np.float32(0.5), _sigm(bb))
                self.hb[k] = ((np.float32(1.0) - z) * self.hb[k] + z * g).astype(np.float32)
                u = u + (self.hb[k] @ W[p + "wout"] + W[p + "bout"])
                if self.m["lrnn_mlp"]:
                    u = u + (_gelu(_rms(u, W[p + "n2"]) @ W[p + "w1"] + W[p + "b1"]) @ W[p + "w2"] + W[p + "b2"])
            u = _rms(u, W["norm_f"])
        y = u @ W["head_w"] + W["head_b"]
        if self.bound > 0:
            y = self.bound * np.tanh(y / self.bound)
        return y.astype(np.float32)


class CNet:
    """The same step in C (kernel.c), compiled for this CPU."""

    def __init__(self, Wt: dict, meta: dict, Wraw: dict | None = None) -> None:
        kind = meta["kind"]
        wf16 = meta.get("weights", "f32") == "f16"
        Wraw = Wraw or Wt
        gl = meta["gru_layers"] if kind in ("gru", "hybrid") else 0
        nb = meta["lrnn_layers"] if kind in ("lrnn", "hybrid") else 0
        if gl > MAX_GRU or nb > MAX_BLK:
            raise ValueError("too many layers for kernel.c")
        names = ["mean", "std", "proj_w", "proj_b"]
        names += [f"gru{l}_{k}" if l < gl else None for l in range(MAX_GRU) for k in ("wi", "bi", "wh", "bh")]
        names += [f"blk{k}_{n}" if k < nb else None for k in range(MAX_BLK) for n in BLK_KEYS]
        names += ["norm_f", "head_w", "head_b"]
        off, chunks, hchunks, pos, hpos = [], [], [], 0, 0
        for n in names:
            if n is None or n not in Wt:
                off.append(-1); continue
            if wf16 and is_matrix(n):
                a = np.ascontiguousarray(Wraw[n], np.float16).ravel().view(np.uint16)
                off.append(hpos); hchunks.append(a); hpos += a.size
            else:
                a = np.ascontiguousarray(Wt[n], np.float32).ravel()
                off.append(pos); chunks.append(a); pos += a.size
        self.Wflat = np.concatenate(chunks).astype(np.float32)
        self.Wh = np.concatenate(hchunks) if hchunks else np.zeros(8, np.uint16)
        self.off = np.array(off, np.int32)
        self.cfg = np.array([KINDS[kind], len(Wt["mean"]), int(meta["diff"]), meta["d_proj"], meta["gru_hidden"],
                             gl, nb, meta["lrnn_state"], meta["lrnn_mlp"], meta["reset_every"], 2, int(wf16)], np.int32)
        self.fcfg = np.array([meta["clip"], meta["out_bound"]], np.float32)
        self.hs = np.zeros(max(1, gl * meta["gru_hidden"] + nb * meta["lrnn_state"]), np.float32)
        self.prev = np.zeros(len(Wt["mean"]), np.float32)
        self.sc = np.zeros(4 * (len(Wt["mean"]) + 4 * max(meta["d_proj"], meta["gru_hidden"])
                                + 6 * meta["gru_hidden"] + 4 * meta["lrnn_state"] + 2 * meta["lrnn_mlp"]) + 1024,
                           np.float32)
        self.inp = np.zeros(len(Wt["mean"]), np.float32)
        self.out = np.zeros(2, np.float32)
        lib = ctypes.CDLL(str(_compile()))
        self.fn = lib.wnn_step
        self.fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.fn.restype = None
        bufs = [self.cfg, self.fcfg, self.off, self.Wflat, self.hs, self.prev, self.sc, self.inp, self.out, self.Wh]
        self.ctx = np.array([b.ctypes.data for b in bufs], np.uint64)
        self.ctx_ptr = self.ctx.ctypes.data

    def reset(self) -> None:
        self.hs[:] = 0.0

    def step(self, raw: np.ndarray, first: bool) -> np.ndarray:
        self.inp[:] = raw
        self.fn(self.ctx_ptr, 1 if first else 0)
        return self.out.copy()


_SO: Path | None = None


def _compile() -> Path:
    global _SO
    if _SO is not None and _SO.exists():
        return _SO
    src = HERE / "kernel.c"
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc:
        raise RuntimeError("no C compiler")
    tag = hashlib.md5(src.read_bytes()).hexdigest()[:12]
    out = Path(tempfile.gettempdir()) / f"wnn_kernel_{tag}_{os.getpid()}.so"
    subprocess.run([cc, "-O3", "-march=native", "-fPIC", "-shared", "-fno-math-errno", str(src),
                    "-o", str(out), "-lm"], check=True, capture_output=True, timeout=120)
    _SO = out
    return out


def _load(i: int = 0):
    """-> (float32 weights for numpy, weights as stored (float16 matrices when weights=f16), meta) of member i"""
    mf, nf = ("meta.json", "model.npz") if i == 0 else (f"meta_{i}.json", f"model_{i}.npz")
    meta = json.loads((HERE / mf).read_text())
    with np.load(HERE / nf) as z:
        Wraw = {k: np.ascontiguousarray(z[k]) for k in z.files}
    Wt = {k: v.astype(np.float32) for k, v in Wraw.items()}
    return Wt, Wraw, meta


class AvgNet:
    """Mean of several networks that see the same rows (an ensemble in one submission)."""

    def __init__(self, nets: list) -> None:
        self.nets = nets
        self.prev = nets[0].prev

    def reset(self) -> None:
        for n in self.nets:
            n.reset()

    def step(self, raw: np.ndarray, first: bool) -> np.ndarray:
        out = self.nets[0].step(raw, first).astype(np.float32)
        for n in self.nets[1:]:
            out += n.step(raw, first)
        return out / np.float32(len(self.nets))


def _wrap(nets: list):
    return nets[0] if len(nets) == 1 else AvgNet(nets)


def make_net(backend: str | None = None):
    """-> (net, backend name). 'auto' tries C and checks every member against numpy on random rows."""
    meta0 = json.loads((HERE / "meta.json").read_text())
    members = [_load(i) for i in range(int(meta0.get("members", 1)))]
    backend = backend or os.environ.get("WNN_BACKEND") or meta0.get("backend", "auto")
    if backend in ("auto", "c"):
        try:
            nets = []
            for Wt, Wraw, meta in members:
                cnet, ref = CNet(Wt, meta, Wraw), NumpyNet(Wt, meta)
                rng = np.random.default_rng(0)
                for t in range(64):
                    raw = rng.standard_normal(len(Wt["mean"])).astype(np.float32) * Wt["std"] + Wt["mean"]
                    a, b = cnet.step(raw, t == 0), ref.step(raw, t == 0)
                    if not np.all(np.isfinite(a)) or np.max(np.abs(a - b)) > 1e-3:
                        raise RuntimeError(f"C step disagrees with numpy by {np.max(np.abs(a - b)):.2e}")
                cnet.reset(); cnet.prev[:] = 0.0
                nets.append(cnet)
            return _wrap(nets), "c"
        except Exception as e:                       # noqa: BLE001 - any failure -> numpy
            if backend == "c":
                raise
            print(f"[wnn] C backend unavailable ({e.__class__.__name__}: {e}); using numpy", flush=True)
    return _wrap([NumpyNet(Wt, meta) for Wt, _, meta in members]), "numpy"


class PredictionModel:
    def __init__(self, backend: str | None = None) -> None:
        self.net, self.backend = make_net(backend)
        meta = json.loads((HERE / "meta.json").read_text())
        self.reset_every = int(meta["reset_every"])
        self.f16 = meta.get("dtype", "float16") == "float16"
        self.buf16 = np.zeros(len(self.net.prev), np.float16)
        self.buf32 = np.zeros(len(self.net.prev), np.float32)
        self.seq = None

    def predict(self, data_point):
        step = data_point.step_in_seq
        first = False
        if data_point.seq_ix != self.seq:
            self.seq = data_point.seq_ix
            self.net.reset()
            first = True
        elif self.reset_every and step % self.reset_every == 0:
            self.net.reset()
        if self.f16:                       # the training store holds float16: round identically
            self.buf16[:] = data_point.state
            self.buf32[:] = self.buf16
            raw = self.buf32
        else:
            raw = np.asarray(data_point.state, np.float32)
        out = self.net.step(raw, first)
        if not data_point.need_prediction:
            return None
        return out


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", required=True)
    ap.add_argument("--starterpack", default=None, help="folder holding utils.py")
    a = ap.parse_args()
    sys.path.insert(0, a.starterpack or str(Path(a.validation).resolve().parents[1]))
    from utils import ScorerStepByStep
    m = PredictionModel()
    print(f"backend={m.backend}")
    print(ScorerStepByStep(a.validation).score(m))
