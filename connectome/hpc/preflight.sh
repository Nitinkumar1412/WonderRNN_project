#!/bin/bash
# Checks before burning a GPU allocation. Safe on the login node.
#   cd ~/wunder_fund_challenge/code/connectome && bash hpc/preflight.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source hpc/_config.sh
activate_env
wnn_need_writable "$CODE_DIR/logs" || exit 1
echo "== wnn preflight (paths / data / packages / GPU) =="
python -m wnn preflight "$@"
echo
echo "== unit tests (CPU) =="
python -m pytest tests -q -p no:cacheprovider
echo
echo "Ready:  bash hpc/run_pipeline.sh smoke    then    bash hpc/run_pipeline.sh extreme"
