"""evaluate: held-out score of a run  (GPU)

    python -m wnn evaluate --run NAME [--fold F]      (default fold: the one the run held out)
    python -m wnn evaluate --runs A,B [--fold F]      (members + mean of outputs, valid-mask WP)

Target runs report global WP with the real validation mask (the competition's
setup), on all required rows, and by quarter of the sequence. Mask runs report
AUC on held-out valid sequences; pretrain runs report next-step MSE.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import SEQ_LEN, WARMUP, Store
from .metric import Moments
from .models import featurize, segment_len
from .runs import device, load_run
from .util import Work, log, save_json

QUARTERS = [(WARMUP, 5000), (5000, 10000), (10000, 15000), (15000, SEQ_LEN)]


@torch.no_grad()
def predict(model, rcfg, st: Store, ids, batch_seqs: int = 16, dev: str | None = None):
    """Yield (seq id, predictions [T, n_out] float32), running each reset-segment from a zero state."""
    dev = dev or next(model.parameters()).device
    L = segment_len(rcfg)
    mean = torch.as_tensor(st.mean, device=dev)
    std = torch.as_tensor(st.std, device=dev)
    ids = list(ids)
    c = 0
    while c < len(ids):
        chunk = ids[c:c + batch_seqs]
        raw = f = None
        try:
            raw = torch.from_numpy(np.stack([np.asarray(st.X[i]) for i in chunk])).to(dev)
            raw = torch.cat([raw[:, :1], raw], dim=1)
            f = featurize(raw, mean, std, float(rcfg.features.clip), bool(rcfg.features.diff))
            n = len(chunk)
            out, _ = model(f.reshape(n * (SEQ_LEN // L), L, f.shape[-1]))
            out = out.reshape(n, SEQ_LEN, -1).float().cpu().numpy()
        except torch.OutOfMemoryError:
            raw = f = None
            torch.cuda.empty_cache()
            if batch_seqs == 1:
                raise
            batch_seqs = max(1, batch_seqs // 2)
            log(f"CUDA OOM in predict -> {batch_seqs} sequences per batch")
            continue
        for k, i in enumerate(chunk):
            yield i, out[k]
        c += len(chunk)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    pos = label.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def score_ids(model, rcfg, st: Store, ids, objective: str, batch_seqs: int = 16) -> dict:
    ids = np.asarray(ids)
    need = np.arange(SEQ_LEN) >= WARMUP
    if objective == "target":
        vmask, allrows = Moments(), Moments()
        quarters = [Moments() for _ in QUARTERS]
        n_valid = 0
        for i, p in predict(model, rcfg, st, ids, batch_seqs):
            y = np.asarray(st.Y[i])
            allrows.add(y, p, need)
            if st.is_valid[i]:
                n_valid += 1
                sel = need & (np.asarray(st.scored[i]) == 1)
                vmask.add(y, p, sel)
                for q, (a, b) in zip(quarters, QUARTERS):
                    s = sel.copy(); s[:a] = False; s[b:] = False
                    q.add(y, p, s)
        res = {"wp_all_rows": allrows.score(), "wp_all_rows_t": allrows.per_target(),
               "n_seqs": int(len(ids)), "n_valid_seqs": n_valid}
        if n_valid:
            res.update({"wp_valid_mask": vmask.score(), "wp_valid_mask_t": vmask.per_target(),
                        "wp_quarters": [q.score() for q in quarters]})
        return res
    if objective == "mask":
        sc, lab = [], []
        for i, p in predict(model, rcfg, st, ids, batch_seqs):
            sc.append(p[need, 0]); lab.append(np.asarray(st.scored[i])[need] == 1)
        return {"auc": auc(np.concatenate(sc), np.concatenate(lab)), "n_seqs": int(len(ids))}
    if objective == "pretrain":
        tot, cnt = 0.0, 0
        for i, p in predict(model, rcfg, st, ids, batch_seqs):
            x = np.clip((np.asarray(st.X[i], np.float32) - st.mean) / st.std,
                        -rcfg.features.clip, rcfg.features.clip)
            tgt = x[1:] - x[:-1]
            tot += float(((p[:-1] - tgt) ** 2).sum()); cnt += tgt.size
        return {"next_dx_mse": tot / max(cnt, 1), "n_seqs": int(len(ids))}
    raise ValueError(objective)


def heldout_subset(st: Store, ids: np.ndarray, max_seqs: int) -> np.ndarray:
    """Valid sequences first (they carry the real mask), then train ones, capped at max_seqs."""
    ids = np.asarray(ids)
    if not max_seqs or len(ids) <= max_seqs:
        return ids
    v, t = ids[st.is_valid[ids]], ids[~st.is_valid[ids]]
    nv = min(len(v), max(max_seqs // 2, max_seqs - len(t)))
    return np.concatenate([v[:nv], t[: max_seqs - nv]])


def ensemble_score(members, st: Store, ids, batch_seqs: int = 16) -> dict:
    """Held-out WP (real valid mask) of each member and of the mean of their outputs."""
    need = np.arange(SEQ_LEN) >= WARMUP
    each = [Moments() for _ in members]
    avg = Moments()
    gens = [predict(m, r, st, ids, batch_seqs) for m, r in members]
    for outs in zip(*gens):
        i = outs[0][0]
        if not st.is_valid[i]:
            continue
        y = np.asarray(st.Y[i])
        sel = need & (np.asarray(st.scored[i]) == 1)
        ps = [o[1] for o in outs]
        for mom, p in zip(each, ps):
            mom.add(y, p, sel)
        avg.add(y, np.mean(ps, axis=0), sel)
    return {"members_wp_valid_mask": [m.score() for m in each], "ensemble_wp_valid_mask": avg.score()}


def run(cfg, args) -> None:
    W = Work(cfg)
    if args.runs:
        names = [n for n in args.runs.split(",") if n]
        members, folds = [], set()
        for n in names:
            model, rcfg, ck = load_run(W.root, n, device())
            members.append((model, rcfg)); folds.add(ck["fold"])
        if args.fold is None and len(folds) > 1:
            raise SystemExit(f"runs held out different folds: {folds}")
        fold = args.fold if args.fold is not None else next(iter(folds))
        st = Store(W.arrays(""), members[0][1].data.dtype)
        _, held = st.split(fold, "target")
        held = heldout_subset(st, held, int(cfg.eval.max_seqs))
        log(f"ensemble {names} on fold {fold}: {len(held)} seqs")
        res = ensemble_score(members, st, held, int(cfg.eval.batch))
        res.update({"runs": names, "fold": int(fold)})
        save_json(res, W.run_dir(names[0]) / f"ens_{len(names)}_fold{fold}.json")
        log(f"result: {res}")
        return
    if not args.run:
        raise SystemExit("evaluate needs --run NAME")
    model, rcfg, ck = load_run(W.root, args.run, device())
    fold = ck["fold"] if args.fold is None else args.fold
    st = Store(W.arrays(""), rcfg.data.dtype)
    _, held = st.split(fold, ck["objective"])
    if len(held) == 0:
        raise SystemExit(f"run {args.run} has no held-out fold (fold={fold}); pass --fold F")
    held = heldout_subset(st, held, int(cfg.eval.max_seqs))
    log(f"evaluating {args.run} ({ck['objective']}, {rcfg.model.kind}) on fold {fold}: {len(held)} seqs")
    res = score_ids(model, rcfg, st, held, ck["objective"], int(cfg.eval.batch))
    res["fold"] = int(fold)
    save_json(res, W.run_dir(args.run) / f"eval_fold{fold}.json")
    log(f"result: {res}")
