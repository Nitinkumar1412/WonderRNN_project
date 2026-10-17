"""teacher: averaged predictions of a zoo of runs -> teachers/<distill.teacher>.f16.npy  (GPU)

    python -m wnn teacher --runs big_f0,big_f1,big_f2,big_f3,big_f4,tf_f0,... distill.teacher=zoo1

Out-of-fold where possible: a sequence of fold k is labelled by the runs that held
fold k out; only when no run held it out are all runs averaged. Each run is rebuilt
from its own config (kind, width, reset period), so teachers can differ freely.
A student then trains with `distill.teacher=zoo1 distill.lam=0.5`.
"""
from __future__ import annotations

import numpy as np

from .data import SEQ_LEN, Store
from .evaluate import predict
from .runs import device, load_run
from .util import Work, as_list, log, save_json, timed


def run(cfg, args) -> None:
    W = Work(cfg)
    names = as_list(args.runs)
    if not names or not cfg.distill.teacher:
        raise SystemExit("teacher needs --runs a,b,c and distill.teacher=<name>")
    st = Store(W.arrays(""), cfg.data.dtype, with_mask_p=False)
    acc = np.zeros((st.n, SEQ_LEN, 2), np.float32) if st.n * SEQ_LEN * 8 < 8e9 else None
    if acc is None:
        acc = np.lib.format.open_memmap(W.p("teachers", "_acc.tmp.npy"), mode="w+", dtype=np.float32,
                                        shape=(st.n, SEQ_LEN, 2))
    cnt = np.zeros(st.n, np.int32)
    folds = {}
    for name in names:
        model, rcfg, ck = load_run(W.root, name, device())
        if ck["objective"] != "target":
            raise SystemExit(f"{name} is not a target run")
        folds[name] = ck["fold"]
    oof_folds = {f for f in folds.values() if f >= 0}
    for name in names:
        model, rcfg, ck = load_run(W.root, name, device())
        f = ck["fold"]
        # this run labels: its own held-out fold, plus every sequence whose fold no run held out
        ids = np.where((st.fold == f) | ~np.isin(st.fold, list(oof_folds)))[0] if f >= 0 else \
            np.where(~np.isin(st.fold, list(oof_folds)))[0]
        with timed(f"{name} (fold {f}) on {len(ids)} seqs"):
            for i, p in predict(model, rcfg, st, ids, int(cfg.eval.batch)):
                acc[i] += p[:, :2]
                cnt[i] += 1
    if (cnt == 0).any():
        raise SystemExit(f"{int((cnt == 0).sum())} sequences got no teacher prediction")
    out = np.lib.format.open_memmap(W.p("teachers", f"{cfg.distill.teacher}.f16.npy"), mode="w+",
                                    dtype=np.float16, shape=(st.n, SEQ_LEN, 2))
    for i in range(st.n):
        out[i] = acc[i] / cnt[i]
    out.flush()
    save_json({"runs": names, "folds": folds, "per_seq_count": np.bincount(cnt).tolist()},
              W.p("teachers", f"{cfg.distill.teacher}.json"))
    log(f"teacher {cfg.distill.teacher}: {len(names)} runs, teachers per sequence {np.bincount(cnt).tolist()}")
