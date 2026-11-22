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
"$PYTHON" -m src.hw2_train --config configs/hw2.json
"$PYTHON" -m src.hw2_report
"$PYTHON" -m src.hw2_predict --output submission_hw2.csv
