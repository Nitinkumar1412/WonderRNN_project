"""Saved runs: runs/<name>/model.pt holds the (EMA) weights and the run's own config,
so later steps rebuild the exact model whatever the current config says."""
from __future__ import annotations

from pathlib import Path

import torch

from .config import from_dict
from .models import build


def save_run(path: Path, state: dict, cfg, objective: str, fold: int, metrics: dict) -> None:
    torch.save({"state": state, "config": dict(cfg), "objective": objective, "fold": int(fold),
                "metrics": metrics}, path)


def load_run(work_root: Path, run: str, device: str = "cpu"):
    """-> (model in eval mode, run cfg, checkpoint dict)."""
    path = Path(work_root) / "runs" / run / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"no trained run at {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    rcfg = from_dict(ck["config"])
    model = build(rcfg, ck["objective"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), rcfg, ck


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
