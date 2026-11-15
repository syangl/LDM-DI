#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# all baselines need to run
BASELINES=("secmi" "piami" "naivelossmi" "rediffusemi" "structuralmi")

echo "Start Running..."

for method in "${BASELINES[@]}"
do
    echo "========================================"
    echo "Current Running Baseline: $method"
    echo "========================================"
    
    python "$SCRIPT_DIR/exp_utils/mi_util/baseline_mi_main.py" --baseline "$method" --is_remove "$@"
    
    # check state
    if [ $? -ne 0 ]; then
        echo "Error: $method running failed"
        exit 1
    fi
done

echo "All baselines done."