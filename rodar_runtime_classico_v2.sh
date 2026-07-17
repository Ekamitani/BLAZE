#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

python vision/cylinders_detect/classic_vision/_runtime_classico_v2_core.py \
    --trained-params \
    --no-controls \
    "$@"
