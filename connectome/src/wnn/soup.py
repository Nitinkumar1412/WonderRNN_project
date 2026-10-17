"""soup: average the weights of several runs into one (zero inference cost)  (CPU/GPU)

    python -m wnn soup --runs ft_s1,ft_s2,ft_s3 --run soup1 [--fold 0]

Works when the runs share an architecture and a starting point (e.g. fine-tunes
of one pretrain run with different seeds / data orders): otherwise weights are
not aligned and the average is meaningless. The soup is scored on --fold (default:
the fold the first run held out) next to each ingredient.
"""
from __future__ import annotations

import torch

from .data import Store
from .evaluate import heldout_subset, score_ids
from .runs import device, load_run, save_run
from .util import Work, as_list, log, save_json


def run(cfg, args) -> None:
    W = Work(cfg)
    names = as_list(args.runs)
    if len(names) < 2 or not args.run:
        raise SystemExit("soup needs --runs a,b[,c...] and --run <output name>")
    dev = device()
    models, states = [], []
    for n in names:
        m, rcfg, ck = load_run(W.root, n, dev)
        models.append((n, m, rcfg, ck)); states.append(m.state_dict())
    keys = states[0].keys()
    for n, s in zip(names, states):
        if s.keys() != keys or any(s[k].shape != states[0][k].shape for k in keys):
            raise SystemExit(f"{n} has a different architecture from {names[0]}")
    avg = {k: (torch.stack([s[k].float() for s in states]).mean(0).to(states[0][k].dtype)
               if states[0][k].dtype.is_floating_point else states[0][k]) for k in keys}
    _, first, rcfg, ck0 = models[0]
    fold = ck0["fold"] if args.fold is None else args.fold
    metrics = {"ingredients": names, "fold": fold}
    if fold is not None and fold >= 0:
        st = Store(W.arrays(""), rcfg.data.dtype, with_mask_p=False)
        _, held = st.split(fold, ck0["objective"])
        held = heldout_subset(st, held, int(cfg.eval.max_seqs))
        obj, eb = ck0["objective"], int(cfg.eval.batch)
        metrics["ingredients_scores"] = {n: score_ids(m, rc, st, held, obj, eb) for n, m, rc, _ in models}
        first.load_state_dict(avg)
        metrics["soup"] = score_ids(first.eval(), rcfg, st, held, obj, eb)
        log(f"soup: {metrics['soup']}")
        for n, s in metrics["ingredients_scores"].items():
            log(f"  {n}: {s}")
    rdir = W.run_dir(args.run)
    save_run(rdir / "model.pt", {k: v.cpu() for k, v in avg.items()}, rcfg, ck0["objective"],
             fold if fold is not None else -1, metrics)
    save_json(metrics, rdir / "metrics.json")
    log(f"saved soup {args.run} from {len(names)} runs")
