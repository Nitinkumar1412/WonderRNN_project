"""Small shared helpers: logging, timing, JSON, thread counts, work paths."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_T0 = time.time()


def log(msg: str) -> None:
    el = time.time() - _T0
    print(f"[wnn {time.strftime('%H:%M:%S')} +{el/60:6.1f}m] {msg}", flush=True)


@contextmanager
def timed(what: str) -> Iterator[None]:
    t = time.time()
    log(f"{what} ...")
    yield
    log(f"{what} done in {time.time() - t:.1f}s")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def as_list(x: Any) -> list[str]:
    """Config value -> list of strings: a YAML/JSON list, or 'a,b' / '[a,b]' from the command line."""
    if x is None or x == "":
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [p.strip().strip("'\"") for p in str(x).strip().strip("[]").split(",") if p.strip()]


def n_threads(requested: int = 0) -> int:
    """Threads to use: explicit value, else the SLURM allocation, else all cores."""
    if requested and requested > 0:
        return int(requested)
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return int(slurm)
    return os.cpu_count() or 1


class Work:
    """Paths under paths.work_dir.

    arrays/     X.f16.npy  Y.f32.npy  scored.u8.npy  mask_p.f16.npy  index.json  stats.npz
    runs/<run>/ config.json  model.pt  metrics.json        (one trained model each)
    teachers/   <name>.f16.npy                              (soft targets, [N, T, 2])
    submissions/<name>/  solution.py  model.npz  meta.json  kernel.c  + <name>.zip
    logs/       resolved config of every step
    """

    def __init__(self, cfg) -> None:
        self.root = Path(cfg.paths.work_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def p(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def arrays(self, name: str = "") -> Path:
        d = self.root / "arrays"
        d.mkdir(parents=True, exist_ok=True)
        return d / name if name else d

    def run_dir(self, run: str) -> Path:
        d = self.root / "runs" / run
        d.mkdir(parents=True, exist_ok=True)
        return d
