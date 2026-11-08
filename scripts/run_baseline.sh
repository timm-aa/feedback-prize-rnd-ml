#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${TMPDIR:-/tmp}/feedback-prize-matplotlib"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" -m src.train --config configs/baseline.json
"$PYTHON" -m src.report
"$PYTHON" -m src.predict
