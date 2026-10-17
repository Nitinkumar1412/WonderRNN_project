#!/bin/bash
# One-time environment setup: builds $WNN_ROOT/envs/wnn (python 3.11, torch, requirements.txt;
# ~5 GB). Needs internet - where compute nodes are offline, run it on the login node.
#
#   bash hpc/setup_env.sh
#
# torch: the CUDA build follows this node's NVIDIA driver (nvidia-smi "CUDA Version"); with no
# GPU on this node (a login node) cu118 where glibc < 2.28, else the newest torch on PyPI.
# Your own choice:  WNN_TORCH_SPEC="torch==2.6.0+cu124" WNN_TORCH_INDEX=https://download.pytorch.org/whl/cu124
#
# Base python, first found wins:
#   1. WNN_PYTHON=/path/to/python3.11       venv from that interpreter
#   2. conda from a module or on PATH       new env, python 3.11 from conda-forge
#   3. python >= 3.10 on PATH
#   4. Miniforge installed into $WNN_ROOT/miniforge3
# Re-running is safe: it only installs what is missing.

set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_here/_config.sh"
REQ_FILE="$CODE_DIR/requirements.txt"

if ! wnn_online https://pypi.org/simple/pip/; then
    echo "ERROR: $(hostname) cannot reach pypi.org. Run this on the login node (or set https_proxy)." >&2
    exit 1
fi
wnn_need_writable "$(dirname "$ENV_DIR")" || exit 1

py_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; }
first_python() {
    local c p
    for c in python3.12 python3.11 python3.10 python3 python; do
        p="$(command -v "$c" 2>/dev/null)" || continue
        py_ok "$p" && { echo "$p"; return 0; }
    done
    return 1
}

if [ -e "$ENV_DIR" ] && ! env_python_runs; then
    mv "$ENV_DIR" "$ENV_DIR.stale.$(date +%Y%m%d_%H%M%S)"
fi

if [ ! -e "$ENV_DIR" ]; then
    how="" PY=""
    if [ -n "${WNN_PYTHON:-}" ] && py_ok "$WNN_PYTHON"; then
        how="venv" PY="$WNN_PYTHON"
    else
        for m in ${WNN_CONDA_MODULE:-} anaconda3 Anaconda3 miniconda3 Miniconda3 miniforge3 Miniforge3 conda; do
            if _wnn_mload "$m" && command -v conda >/dev/null 2>&1; then how="conda"; break; fi
        done
        [ -z "$how" ] && command -v conda >/dev/null 2>&1 && how="conda"
        [ -z "$how" ] && PY="$(first_python)" && how="venv"
        if [ -z "$how" ]; then
            dir="$WNN_ROOT/miniforge3"
            curl -fL -o /tmp/mf.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
            bash /tmp/mf.sh -b -p "$dir" && rm -f /tmp/mf.sh
            export PATH="$dir/bin:$PATH"; how="conda"
        fi
    fi
    case "$how" in
        venv)  echo ">> venv from $PY ($("$PY" --version 2>&1))"; "$PY" -m venv "$ENV_DIR" ;;
        conda) echo ">> conda env (python 3.11, conda-forge)"
               conda create -y --quiet -p "$ENV_DIR" --override-channels -c conda-forge python=3.11 pip ;;
    esac
    : > "$ENV_DIR/.wnn_module"
fi

enter_env
export PIP_PREFER_BINARY=1
python -m pip install --upgrade pip >/dev/null 2>&1 || true

if python -c "import torch" 2>/dev/null; then
    echo ">> torch $(python -c 'import torch; print(torch.__version__)') already installed"
elif [ -n "${WNN_TORCH_SPEC:-}${WNN_TORCH_INDEX:-}" ]; then
    python -m pip install --index-url "${WNN_TORCH_INDEX:-https://pypi.org/simple}" "${WNN_TORCH_SPEC:-torch}"
else
    DRV_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -n 1 || true)"
    GLIBC="$(python -c 'import platform; print(platform.libc_ver()[1] or "0")')"
    BUILDS="$(python - "$DRV_CUDA" "$GLIBC" <<'PY'
import sys
def ver(s):
    p = [int(x) for x in s.split(".")[:2] if x.isdigit()]
    return tuple(p + [0] * (2 - len(p)))
drv, glibc = sys.argv[1], ver(sys.argv[2])
builds = [((13, 0), "cu130"), ((12, 8), "cu128"), ((12, 6), "cu126"), ((12, 4), "cu124"), ((12, 1), "cu121"), ((11, 8), "cu118")]
print(" ".join(t for need, t in builds if ver(drv) >= need) if drv else ("cu118" if glibc < (2, 28) else "pypi"))
PY
)"
    echo ">> driver CUDA ${DRV_CUDA:-none}, glibc $GLIBC -> torch builds to try: $BUILDS"
    got=0
    for b in $BUILDS; do
        idx="https://download.pytorch.org/whl/$b"; [ "$b" = pypi ] && idx="https://pypi.org/simple"
        if python -m pip install --index-url "$idx" torch; then got=1; break; fi
    done
    [ "$got" = 1 ] || { echo "ERROR: no torch build installs here - set WNN_TORCH_SPEC / WNN_TORCH_INDEX" >&2; exit 1; }
fi

while IFS= read -r line || [ -n "$line" ]; do
    req="$(printf '%s' "${line%%#*}" | tr -d '[:space:]')"
    [ -z "$req" ] && continue
    case "$req" in torch*) continue ;; esac
    python -m pip install "$req" || python -m pip install "$(printf '%s' "$req" | sed -E 's/[<>=!~].*//')"
done < "$REQ_FILE"
python -m pip freeze > "$CODE_DIR/requirements.lock.txt"

python - <<'PY'
import importlib, sys
miss = []
for m in ["torch", "numpy", "pyarrow", "yaml"]:
    try:
        x = importlib.import_module(m); print(f"  {m:8s} {getattr(x, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {m:8s} MISSING ({e})"); miss.append(m)
import torch
print(f"  cuda     {torch.version.cuda} available={torch.cuda.is_available()}"
      + (f" on {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else " (normal on a login node)"))
sys.exit("ENV NOT READY: " + ", ".join(miss) if miss else 0)
PY
echo "ENV READY at $ENV_DIR"
