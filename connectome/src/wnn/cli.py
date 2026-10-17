"""Command line: one sub-command per pipeline step.

    python -m wnn <step> [--config configs/default.yaml] [--fold F] [--run NAME] [key=value ...]

Steps (see README.md):
    preflight    paths, data files, packages, GPU                                    (CPU)
    prepare      parquet -> memory-mapped arrays, feature stats, folds               (CPU)
    train        train one model; objective from train.objective (target|mask|pretrain)  (GPU)
                 --fold F holds fold F out (-1 = train on everything)
    label-mask   run a mask model over every sequence -> arrays/mask_p.f16.npy       (GPU)
    teacher      average runs' predictions -> teachers/<distill.teacher>.f16.npy     (GPU)
    soup         average the weights of runs (same architecture) -> a new run        (CPU)
    evaluate     held-out global WP of a run: valid mask, all rows, by quarter      (GPU)
    export       submission folder + zip, parity check against torch, latency bench (CPU)
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time

from .config import DEFAULT_CONFIG, load_config, to_json
from .util import Work, log

STEPS = {
    "preflight": "wnn.preflight",
    "prepare": "wnn.prepare",
    "train": "wnn.train",
    "label-mask": "wnn.mask",
    "teacher": "wnn.teacher",
    "soup": "wnn.soup",
    "evaluate": "wnn.evaluate",
    "export": "wnn.export",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wnn", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=list(STEPS))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--fold", type=int, default=None, help="held-out fold (-1 = none)")
    ap.add_argument("--run", default=None, help="run name (train: output, others: input)")
    ap.add_argument("--runs", default=None, help="comma-separated run names (teacher, soup)")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. train.epochs=2")
    args = ap.parse_intermixed_args(argv)

    cfg = load_config(args.config, args.overrides)
    W = Work(cfg)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    W.p("logs", f"{args.step}_{stamp}.json").write_text(to_json(cfg), encoding="utf-8")
    log(f"step={args.step} run={args.run} fold={args.fold} overrides={args.overrides} work={W.root}")

    module = importlib.import_module(STEPS[args.step])
    module.run(cfg, args)
    log(f"step={args.step} finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
