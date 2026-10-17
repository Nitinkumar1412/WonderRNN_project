"""train: one model, one fold  (GPU)

    python -m wnn train --run gru_r2k_f0 --fold 0                               # target model, fold 0 held out
    python -m wnn train --run lrnn_f0 --fold 0 model.kind=lrnn train.batch=128
    python -m wnn train --run mask_v1 --fold 0 train.objective=mask model.gru_hidden=96 model.aux_horizons=[]
    python -m wnn train --run pre_lrnn --fold -1 train.objective=pretrain model.kind=lrnn
    python -m wnn train --run lrnn_ft_f0 --fold 0 model.kind=lrnn train.init_from=pre_lrnn
    python -m wnn train --run stu_f0 --fold 0 distill.teacher=zoo1

Objectives:
    target    global weighted Pearson (+ weighted MSE, + aux future-target heads), rows
              weighted by |y| and P(scored)
    mask      BCE on is_scored, valid sequences only (feeds label-mask)
    pretrain  next-step dx regression on every sequence (self-supervised; init for target runs)

Saves runs/<run>/model.pt (EMA weights + config) and metrics.json; runs/<run>/last.pt
lets a re-submitted job resume after the last finished epoch.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .batches import SegmentBatches
from .data import Store
from .evaluate import heldout_subset, score_ids
from .metric import GlobalPearsonLoss, weighted_mse
from .models import build, count_step_macs, featurize, load_backbone, segment_len
from .runs import device, load_run, save_run
from .util import Work, log, save_json


def lr_at(step: int, total: int, cfg) -> float:
    t = cfg.train
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps
    prog = min(1.0, (step - t.warmup_steps) / max(1, total - t.warmup_steps))
    return t.lr * (t.min_lr_frac + (1 - t.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def param_groups(model, wd: float):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    other = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return [{"params": decay, "weight_decay": wd}, {"params": other, "weight_decay": 0.0}]


class EMA:
    def __init__(self, model, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def state(self, model) -> dict:
        own = model.state_dict()
        return {k: self.shadow[k].to(own[k].dtype) for k in own}


def targets(cfg, b, L):
    """(training target, true target) - the training target is blended with the teacher when distilling."""
    y_true = b["y"][:, :L]
    if "soft" in b:
        lam = float(cfg.distill.lam)
        return lam * y_true + (1 - lam) * b["soft"], y_true
    return y_true, y_true


def aux_loss(cfg, aux, b, L):
    hs = cfg.model.aux_horizons or []
    if aux is None or not hs:
        return None
    tot = 0.0
    for j, h in enumerate(hs):
        ya = b["y"][:, h:h + L].clamp(-2, 2)
        wa = ya.abs() * (b["w"] * b["yok"][:, h:h + L]).unsqueeze(-1)
        tot = tot + weighted_mse(aux[..., 2 * j:2 * j + 2], ya, wa)
    return float(cfg.loss.aux_weight) * tot


def compute_loss(objective, cfg, out, aux, b, f, ploss: GlobalPearsonLoss):
    L = out.shape[1]
    w = b["w"]
    if objective == "target":
        y, y_true = targets(cfg, b, L)
        loss, corr = ploss(out, y, w, wy=y_true)
        a = aux_loss(cfg, aux, b, L)
        if a is not None:
            loss = loss + a
        return loss, {"corr_t0": float(corr[0]), "corr_t1": float(corr[1])}
    if objective == "mask":
        bce = F.binary_cross_entropy_with_logits(out[..., 0], b["scored"], reduction="none")
        loss = (bce * w).sum() / w.sum().clamp_min(1.0)
        return loss, {}
    if objective == "pretrain":
        x = f[..., : out.shape[-1]]
        tgt = x[:, 1:] - x[:, :-1]
        loss = float(cfg.loss.pretrain_weight) * F.mse_loss(out[:, :-1].float(), tgt.float())
        return loss, {}
    raise ValueError(objective)


def train_step(objective, cfg, fwd, b, feat, amp, ploss: GlobalPearsonLoss, micro: int):
    """Forward + backward of one batch, split into micro-batches of `micro` segments (0 = whole).

    Target objective with micro-batches: a no-grad pass collects the whole batch's Pearson sums,
    then each micro-batch backpropagates the exact full-batch gradient (GlobalPearsonLoss.grad_coef),
    so the update equals the full-batch one at a fraction of the memory. Other objectives (means)
    weight each micro-batch's loss by its share of the batch."""
    B = b["raw"].shape[0]
    mb = min(micro, B) if micro else B
    slices = [slice(i, min(i + mb, B)) for i in range(0, B, mb)]

    def part(s):
        return {k: v[s] for k, v in b.items()}

    def forward(bb):
        f = feat(bb["raw"])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out, aux = fwd(f)
        return f, out, aux

    if objective == "target" and len(slices) > 1:
        with torch.no_grad():
            sums = None
            for s in slices:
                bb = part(s)
                _, out, _ = forward(bb)
                y, yt = targets(cfg, bb, out.shape[1])
                m = ploss.moments(out, y, bb["w"], yt)
                sums = m if sums is None else sums + m
        tot, share = ploss.total(sums)
        for s in slices:
            bb = part(s)
            _, out, aux = forward(bb)
            L = out.shape[1]
            y, yt = targets(cfg, bb, L)
            loss = (ploss.grad_coef(out, y, bb["w"], yt, tot, share, sums[:, 0].sum()) * out.float()).sum()
            a = aux_loss(cfg, aux, bb, L)
            if a is not None:
                loss = loss + a * (s.stop - s.start) / B
            loss.backward()
        corr = ploss.corr_of(tot)
        return float(-(corr / share.float()).mean()), {"corr_t0": float(corr[0]), "corr_t1": float(corr[1])}

    total, extra = 0.0, {}
    for s in slices:
        bb = part(s)
        f, out, aux = forward(bb)
        loss, extra = compute_loss(objective, cfg, out, aux, bb, f, ploss)
        frac = (s.stop - s.start) / B
        (loss * frac).backward()
        total += float(loss) * frac
    return total, extra


def run(cfg, args) -> None:
    W = Work(cfg)
    objective = cfg.train.objective
    fold = -1 if args.fold is None else int(args.fold)
    run_name = args.run or f"{objective}_{cfg.model.kind}_f{fold}_{time.strftime('%m%d_%H%M')}"
    rdir = W.run_dir(run_name)
    dev = device()
    torch.manual_seed(cfg.seed + fold)
    np.random.seed((cfg.seed + fold) % 2**32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    st = Store(W.arrays(""), cfg.data.dtype)
    train_ids, held = st.split(fold, objective)
    if objective == "target" and cfg.loss.mask_source == "true_only":
        train_ids = train_ids[st.is_valid[train_ids]]
    if objective == "target" and cfg.loss.mask_source == "auto" and st.mask_p is None:
        log("WARN: arrays/mask_p.f16.npy missing - train sequences get weight floor+... = 1 (run label-mask)")
    teacher_path = ""
    if objective == "target" and cfg.distill.teacher:
        teacher_path = str(W.p("teachers", f"{cfg.distill.teacher}.f16.npy"))
    L = segment_len(cfg)
    eval_ids = heldout_subset(st, held, int(cfg.train.eval_max_seqs))

    model = build(cfg, objective).to(dev)
    if cfg.train.init_from:
        src, _, _ = load_run(W.root, cfg.train.init_from)
        keys = load_backbone(model, src.state_dict())
        log(f"init_from {cfg.train.init_from}: loaded {len(keys)} tensors")
    n_par = sum(p.numel() for p in model.parameters())
    log(f"run={run_name} objective={objective} kind={cfg.model.kind} params={n_par/1e6:.2f}M "
        f"step MACs={count_step_macs(cfg)/1e3:.0f}k seg_len={L} train_seqs={len(train_ids)} "
        f"heldout={len(held)} device={dev}")

    opt = torch.optim.AdamW(param_groups(model, cfg.train.weight_decay), lr=cfg.train.lr,
                            betas=(0.9, 0.98), fused=(dev == "cuda"))
    ema = EMA(model, cfg.train.ema) if cfg.train.ema > 0 else None
    ploss = GlobalPearsonLoss(cfg.loss.moment_ema, cfg.loss.mse_weight)
    fwd = torch.compile(model) if (cfg.train.compile and dev == "cuda") else model
    amp = dev == "cuda" and cfg.train.amp == "bf16"

    def loader(epoch: int):
        ds = SegmentBatches(str(W.arrays("")), cfg.data.dtype, train_ids, L, int(cfg.train.batch),
                            int(cfg.seed + fold), epoch, objective, cfg.loss.mask_source,
                            float(cfg.loss.mask_floor),
                            list(cfg.model.aux_horizons or []) if objective == "target" else [],
                            teacher_path, int(cfg.train.max_steps))
        nw = int(cfg.train.workers)
        dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=nw, pin_memory=(dev == "cuda"),
                                         prefetch_factor=4 if nw else None)
        return dl, len(ds)

    steps_per_epoch = loader(0)[1]
    total = steps_per_epoch * int(cfg.train.epochs)
    if cfg.train.max_steps:
        total = min(total, int(cfg.train.max_steps))
    step, start_epoch, history = 0, 0, []
    last = rdir / "last.pt"
    if cfg.train.resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if ema and ck.get("ema"):
            ema.shadow = {k: v.to(dev) for k, v in ck["ema"].items()}
        step, start_epoch, history = ck["step"], ck["epoch"] + 1, ck["history"]
        ploss.run = ck.get("ploss_run")
        log(f"resumed from {last}: epoch {start_epoch}, step {step}")
    if ema:
        ema.shadow = {k: v.to(dev) for k, v in ema.shadow.items()}

    mean = torch.as_tensor(st.mean, device=dev)
    std = torch.as_tensor(st.std, device=dev)
    clip, diff = float(cfg.features.clip), bool(cfg.features.diff)
    log_every = int(cfg.train.get("log_every", 50))
    model.ckpt = bool(cfg.train.checkpoint)
    micro = int(cfg.train.micro_batch)

    def feat(raw):
        return featurize(raw, mean, std, clip, diff)

    if dev == "cuda":
        free, tot_mem = torch.cuda.mem_get_info()
        log(f"GPU memory: {free/1e9:.1f} of {tot_mem/1e9:.1f} GB free"
            + ("  (another process holds most of it: batches will be split as needed)" if free < 0.3 * tot_mem else ""))

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        dl, n_b = loader(epoch)
        model.train()
        t0, rows, run_loss = time.time(), 0, []
        for b in dl:
            if cfg.train.max_steps and step >= cfg.train.max_steps:
                break
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, cfg)
            while True:      # out of GPU memory -> checkpoint activations, then halve the micro-batch, retry
                run_backup = ploss.run
                try:
                    opt.zero_grad(set_to_none=True)
                    loss, extra = train_step(objective, cfg, fwd, b, feat, amp, ploss, micro)
                    break
                except torch.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    ploss.run = run_backup
                    torch.cuda.empty_cache()
                    free = torch.cuda.mem_get_info()[0] / 1e9
                    if not model.ckpt:
                        model.ckpt = True
                        log(f"CUDA OOM ({free:.1f} GB free) -> activation checkpointing on")
                    elif (micro or b["raw"].shape[0]) > 1:
                        micro = max(1, (micro or b["raw"].shape[0]) // 2)
                        log(f"CUDA OOM ({free:.1f} GB free) -> micro_batch {micro}")
                    else:
                        raise
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            if ema:
                ema.update(model, step)
            step += 1
            rows += b["raw"].shape[0] * (b["raw"].shape[1] - 1)
            run_loss.append(float(loss))
            if step % log_every == 0:
                el = time.time() - t0
                log(f"ep {epoch} step {step}/{total} loss {np.mean(run_loss[-log_every:]):.4f} "
                    f"gn {float(gn):.2f} lr {opt.param_groups[0]['lr']:.2e} "
                    f"{rows/el/1e6:.2f}M rows/s " + " ".join(f"{k} {v:.4f}" for k, v in extra.items()))
        rec = {"epoch": epoch, "step": step, "train_loss": float(np.mean(run_loss)) if run_loss else None,
               "time_s": time.time() - t0}
        if cfg.train.eval_every_epoch and len(eval_ids):
            emodel = copy.deepcopy(model)
            if ema:
                emodel.load_state_dict(ema.state(model))
            rec.update(score_ids(emodel.eval(), cfg, st, eval_ids, objective, int(cfg.eval.batch)))
            del emodel
        history.append(rec)
        log(f"epoch {epoch} done: {rec}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "ema": ema.shadow if ema else None, "step": step, "epoch": epoch,
                    "history": history, "ploss_run": ploss.run}, last)
        if cfg.train.max_steps and step >= cfg.train.max_steps:
            break

    final = ema.state(model) if ema else model.state_dict()
    final = {k: v.detach().cpu() for k, v in final.items()}
    metrics = {"history": history, "fold": fold, "params": n_par, "step_macs": count_step_macs(cfg)}
    if len(held):
        model.load_state_dict(final)
        ev = heldout_subset(st, held, int(cfg.eval.max_seqs))
        metrics["final"] = score_ids(model.eval(), cfg, st, ev, objective, int(cfg.eval.batch))
        log(f"final held-out ({len(ev)} seqs): {metrics['final']}")
    save_run(rdir / "model.pt", final, cfg, objective, fold, metrics)
    save_json(metrics, rdir / "metrics.json")
    log(f"saved {rdir / 'model.pt'}")
