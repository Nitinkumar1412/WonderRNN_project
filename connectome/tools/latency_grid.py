"""Write random-weight submissions for a grid of model sizes, to time them in the scorer replica
before training anything:

    PYTHONPATH=src python tools/latency_grid.py <out dir> [f16|f32]
    for d in <out dir>/*/; do bash tools/docker_check.sh "$d" 1; done

Latency depends only on the shapes, so random weights time exactly like trained ones.
Edit GRID to try other shapes (same key=value overrides as the CLI).
"""
from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wnn.config import load_config                     # noqa: E402
from wnn.export import INFERENCE_DIR, to_npz           # noqa: E402
from wnn.models import build, count_step_macs          # noqa: E402

GRID = {
    "baseline_gru128x2": ["model.kind=gru", "model.d_proj=128", "model.gru_hidden=128", "model.gru_layers=2"],
    "safe_gru192x2": ["model.kind=gru", "model.d_proj=128", "model.gru_hidden=192", "model.gru_layers=2"],
    "gru256x2": ["model.kind=gru", "model.d_proj=128", "model.gru_hidden=256", "model.gru_layers=2"],
    "stu_lrnn_d96_s384": ["model.kind=lrnn", "model.d_proj=96", "model.lrnn_layers=3", "model.lrnn_state=384",
                          "model.lrnn_mlp=192"],
    "stu_hyb_d128": ["model.kind=hybrid", "model.d_proj=128", "model.gru_hidden=128", "model.gru_layers=1",
                     "model.lrnn_layers=2", "model.lrnn_state=384", "model.lrnn_mlp=0"],
    "floor_lrnn_d16_s16": ["model.kind=lrnn", "model.d_proj=16", "model.lrnn_layers=1", "model.lrnn_state=16",
                           "model.lrnn_mlp=0"],
    "lrnn_d64_s256_x3_m128": ["model.kind=lrnn", "model.d_proj=64", "model.lrnn_layers=3", "model.lrnn_state=256",
                              "model.lrnn_mlp=128"],
    "lrnn_d64_s256_x2_m0": ["model.kind=lrnn", "model.d_proj=64", "model.lrnn_layers=2", "model.lrnn_state=256",
                            "model.lrnn_mlp=0"],
    "lrnn_d96_s192_x2_m96": ["model.kind=lrnn", "model.d_proj=96", "model.lrnn_layers=2", "model.lrnn_state=192",
                             "model.lrnn_mlp=96"],
    "lrnn_d64_s1024": ["model.kind=lrnn", "model.d_proj=64", "model.lrnn_layers=4", "model.lrnn_state=1024",
                       "model.lrnn_mlp=128"],
    "lrnn_d128_s512_x4": ["model.kind=lrnn", "model.d_proj=128", "model.lrnn_layers=4", "model.lrnn_state=512",
                          "model.lrnn_mlp=256"],
}


def main() -> None:
    out = Path(sys.argv[1])
    weights = sys.argv[2] if len(sys.argv) > 2 else "f16"
    st = types.SimpleNamespace(mean=np.zeros(112, np.float32), std=np.ones(112, np.float32))
    for name, ov in GRID.items():
        cfg = load_config(None, ov)
        torch.manual_seed(0)
        net = build(cfg).eval()
        for p in net.parameters():
            p.data.normal_(0, 0.05)
        Wt, meta = to_npz(net, cfg, st, weights)
        meta["backend"] = "auto"
        d = out / f"{name}_{weights}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        np.savez(d / "model.npz", **Wt)
        (d / "meta.json").write_text(json.dumps(meta))
        shutil.copy(INFERENCE_DIR / "solution.py", d)
        shutil.copy(INFERENCE_DIR / "kernel.c", d)
        n_w = sum(v.size for v in Wt.values())
        print(f"{name:22s} {count_step_macs(cfg)/1e3:7.0f}k MACs/step  {n_w*4/1e6:5.2f} MB fp32 weights -> {d}")


if __name__ == "__main__":
    main()
