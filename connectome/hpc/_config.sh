#!/bin/bash
# Single source of truth for cluster paths. Sourced by every hpc/*.sh and by
# run_connectome.sh. Paths derive from where this file lives, so the folder works
# wherever it is uploaded (for example ~/wunder_fund_challenge).
# Override any of them by exporting it first, e.g.
#   WNN_WORK_DIR=/scratch/$USER/wnn_work bash hpc/run_pipeline.sh extreme
#
# Layout (same relative layout as the laptop):
#   $WNN_ROOT/code/connectome                      this package  (CODE_DIR)
#   $WNN_ROOT/wnn_connectome_starterpack           starter pack: datasets/, utils.py
#   $WNN_ROOT/work                                 arrays, runs, teachers, submissions
#   $WNN_ROOT/envs/wnn                             python env (hpc/setup_env.sh)

_wnn_hpc_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${WNN_CODE_DIR:-$(dirname "$_wnn_hpc_dir")}"
WNN_ROOT="${WNN_ROOT:-$(cd "$CODE_DIR/../.." && pwd)}"
ENV_DIR="${WNN_ENV_DIR:-$WNN_ROOT/envs/wnn}"
export WNN_ROOT
# Intermediate arrays are ~60 GB: prefer node-local scratch when the cluster has it.
if [ -z "${WNN_WORK_DIR:-}" ] && [ -d "/scratch/$USER" ] && [ -w "/scratch/$USER" ]; then
    export WNN_WORK_DIR="/scratch/$USER/wnn_work"
fi

# quiet `module load` that also works where Lmod/`module` is absent
_wnn_mload() { type module >/dev/null 2>&1 && module load "$1" >/dev/null 2>&1; }

# Put $ENV_DIR's python first on PATH (conda env or venv); loads the module the env
# was built from (recorded by setup_env.sh), if any.
enter_env() {
    local m=""
    [ -f "$ENV_DIR/.wnn_module" ] && m="$(cat "$ENV_DIR/.wnn_module")"
    if [ -n "$m" ] && ! _wnn_mload "$m"; then
        echo "WARN: could not load module '$m' recorded for $ENV_DIR" >&2
    fi
    export VIRTUAL_ENV="$ENV_DIR"
    export PATH="$ENV_DIR/bin:$PATH"
    # module-provided PYTHONPATH and ~/.local site-packages can shadow the env's packages
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1
}

env_python_runs() {
    ( enter_env >/dev/null 2>&1; [ -x "$ENV_DIR/bin/python" ] && "$ENV_DIR/bin/python" -c "import sys" >/dev/null 2>&1 )
}

# The env has everything the pipeline imports (setup_env.sh then has nothing to do).
wnn_env_ready() {
    ( enter_env >/dev/null 2>&1; [ -x "$ENV_DIR/bin/python" ] &&
      "$ENV_DIR/bin/python" -c "import torch, numpy, pyarrow, yaml" >/dev/null 2>&1 )
}

wnn_online() {
    if command -v curl >/dev/null 2>&1; then
        curl -sfL -o /dev/null -m 20 "$1" >/dev/null 2>&1
    else
        python3 -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=20)" "$1" >/dev/null 2>&1
    fi
}

# Stop with the reason and the fix when a folder the pipeline writes to is not writable.
wnn_need_writable() {
    local d p err rc=0
    for d in "$@"; do
        case "$d" in /*) ;; *) d="$PWD/$d" ;; esac
        if err="$(mkdir -p "$d" 2>&1)" && [ -w "$d" ]; then continue; fi
        p="$d"; while [ ! -d "$p" ]; do p="$(dirname "$p")"; done
        {
            echo "ERROR: cannot write to $d"
            [ -n "$err" ] && echo "  $err"
            echo "  you are '$(id -un)'; the folder in the way is:  $(ls -ld "$p" 2>&1)"
            echo "  If it is yours:  chmod -R u+w $p   - otherwise copy the project into a folder you own."
        } >&2
        rc=1
    done
    return "$rc"
}

activate_env() {
    if ! env_python_runs; then
        echo "ERROR: no working python env at $ENV_DIR" >&2
        echo "Build it once:  bash $CODE_DIR/hpc/setup_env.sh" >&2
        exit 1
    fi
    enter_env
    export PYTHONPATH="$CODE_DIR/src"
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1   # logs arrive at once and in order
    cd "$CODE_DIR" || exit 1
    mkdir -p logs 2>/dev/null || true
}
