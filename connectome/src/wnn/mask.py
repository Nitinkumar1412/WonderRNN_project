"""label-mask: P(is_scored) for every row of every sequence -> arrays/mask_p.f16.npy  (GPU)

    python -m wnn label-mask [--run mask_v1]          (default: mask.run)

The mask model is trained with `train train.objective=mask` (valid sequences only).
Target runs then weight each row by floor + (1 - floor) * P(scored) on train
sequences (valid sequences keep their true mask, see loss.mask_source).
"""
from __future__ import annotations

import numpy as np

from .data import SEQ_LEN, Store
from .evaluate import predict
from .runs import device, load_run
from .util import Work, log, timed


def run(cfg, args) -> None:
    W = Work(cfg)
    name = args.run or cfg.mask.run
    model, rcfg, ck = load_run(W.root, name, device())
    if ck["objective"] != "mask":
        raise SystemExit(f"run {name} is a {ck['objective']} run, not a mask run")
    st = Store(W.arrays(""), rcfg.data.dtype, with_mask_p=False)
    tmp = W.arrays("mask_p.tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(st.n, SEQ_LEN))
    with timed(f"labelling {st.n} sequences with {name}"):
        for k, (i, p) in enumerate(predict(model, rcfg, st, range(st.n), int(cfg.eval.batch))):
            out[i] = 1.0 / (1.0 + np.exp(-p[:, 0]))
            if (k + 1) % 1000 == 0:
                log(f"  {k + 1}/{st.n}")
    out.flush(); del out
    tmp.replace(W.arrays("mask_p.f16.npy"))
    mp = np.load(W.arrays("mask_p.f16.npy"), mmap_mode="r")
    v = np.where(st.is_valid)[0][:50]
    if len(v):
        sc = np.stack([np.asarray(st.scored[i]) for i in v]) == 1
        pv = np.stack([np.asarray(mp[i], np.float32) for i in v])
        log(f"check on {len(v)} valid seqs: mean P = {pv.mean():.3f}, true rate = {sc.mean():.3f}")
