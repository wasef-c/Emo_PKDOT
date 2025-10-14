#!/bin/bash
# Quick start script for PKDOT baseline experiments

set -e  # Exit on error

echo "=========================================="
echo "PKDOT Baseline Experiments"
echo "=========================================="
echo ""

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Default values
CONFIG=${1:-"configs/pkd_baseline.yaml"}
MODE=${2:-"full"}

echo "Configuration: $CONFIG"
echo "Mode: $MODE"
echo ""

# Create checkpoints directory
mkdir -p checkpoints

# Run experiments
if [ "$MODE" == "all" ]; then
    echo "Running ALL experiments from config..."
    python main.py --config "$CONFIG" --all
elif [ "$MODE" == "full" ]; then
    echo "Running full pipeline (teachers + student)..."
    python main.py --config "$CONFIG"
elif [ "$MODE" == "teachers_only" ]; then
    echo "Running teachers only..."
    python main.py --config "$CONFIG" --mode teachers_only
elif [ "$MODE" == "test_single" ]; then
    echo "Testing single experiment (IEMO, seed 42)..."
    python main.py \
        --mode full \
        --dataset IEMO \
        --experiment PKDOT_test \
        --seed 42
else
    echo "Unknown mode: $MODE"
    echo "Usage: ./run_baseline.sh [config_path] [mode]"
    echo "Modes: all, full, teachers_only, test_single"
    exit 1
fi

echo ""
echo "=========================================="
echo "Experiments complete!"
echo "=========================================="
echo ""
echo "Check results in:"
echo "  - checkpoints/ (model checkpoints)"
echo "  - WandB dashboard (logs)"
