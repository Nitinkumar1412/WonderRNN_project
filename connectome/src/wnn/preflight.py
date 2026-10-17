"""preflight: paths, data, packages, GPU - before burning an allocation  (CPU)"""
from __future__ import annotations

import importlib
import shutil
from pathlib import Path

from .data import x_name
from .util import Work, log


def run(cfg, args) -> None:
    ok = True
    W = Work(cfg)
    d = Path(cfg.paths.data_dir)
    for f in ("train.parquet", "valid.parquet"):
        p = d / f
        log(f"{'ok ' if p.exists() else 'MISSING'} {p}" + (f" ({p.stat().st_size/1e9:.1f} GB)" if p.exists() else ""))
        ok &= p.exists()
    sp = Path(cfg.paths.starterpack) / "utils.py"
    log(f"{'ok ' if sp.exists() else 'missing (export uses its own DataPoint)'} {sp}")
    for m in ("torch", "numpy", "pyarrow", "yaml"):
        try:
            x = importlib.import_module(m)
            log(f"ok  {m} {getattr(x, '__version__', '')}")
        except Exception as e:                  # noqa: BLE001
            log(f"MISSING {m} ({e})"); ok = False
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log(f"ok  cuda {torch.version.cuda} on {p.name} ({p.total_memory/1e9:.0f} GB), bf16={torch.cuda.is_bf16_supported()}")
    else:
        log("no GPU visible (fine on a login node; train/teacher/label-mask need one)")
    log(f"{'ok ' if shutil.which('gcc') else 'no '} gcc (export tests the C kernel only where gcc exists)")
    arrays = W.arrays("")
    have = (arrays / x_name(cfg.data.dtype)).exists() and (arrays / "index.json").exists()
    log(f"arrays: {'prepared' if have else 'not prepared yet (run: prepare)'} at {arrays}")
    if have:
        log(f"mask_p: {'present' if (arrays / 'mask_p.f16.npy').exists() else 'absent (run: label-mask)'}")
    total, used, free = shutil.disk_usage(W.root)
    log(f"work dir {W.root}: {free/1e9:.0f} GB free (prepare needs ~60 GB at float16, ~115 GB at float32)")
    if not ok:
        raise SystemExit("preflight failed")
