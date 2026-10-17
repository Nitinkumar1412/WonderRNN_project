#!/bin/bash
# Table of finished runs, best first:  bash hpc/compare.sh [run-name prefix]
# Columns: held-out WP with the real valid mask, all-rows WP, last quarter, MACs/step, run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source hpc/_config.sh
enter_env >/dev/null 2>&1 || true
python - "${WNN_WORK_DIR:-$WNN_ROOT/work}" "${1:-}" <<'PY'
import json, sys
from pathlib import Path
root, prefix = Path(sys.argv[1]) / "runs", sys.argv[2]
rows = []
for m in sorted(root.glob(f"{prefix}*/metrics.json")):
    d = json.loads(m.read_text())
    f = d.get("final") or d.get("soup") or {}
    if "wp_valid_mask" in f or "wp_all_rows" in f:
        rows.append((f.get("wp_valid_mask", float("nan")), f.get("wp_all_rows", float("nan")),
                     (f.get("wp_quarters") or [float("nan")])[-1], d.get("step_macs", 0), m.parent.name, d.get("fold")))
    elif "auc" in f:
        print(f"{m.parent.name:32s} mask AUC {f['auc']:.4f}")
rows.sort(key=lambda r: -r[0] if r[0] == r[0] else 1)
print(f"{'valid_mask':>10s} {'all_rows':>9s} {'last_q':>7s} {'kMACs':>6s}  run (fold)")
for vm, ar, lq, macs, name, fold in rows:
    print(f"{vm:10.4f} {ar:9.4f} {lq:7.4f} {macs/1e3:6.0f}  {name} ({fold})")
PY
