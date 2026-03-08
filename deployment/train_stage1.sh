#!/bin/bash
set -e
echo "=== DataSage Stage 1: Cleaning GRPO Training ==="
echo "GPU Info:"
nvidia-smi
echo ""

echo "=== Installing dependencies ==="
pip install -q unsloth trl transformers accelerate peft bitsandbytes vllm wandb python-dotenv kagglehub
pip install -q "openenv-core[core]>=0.2.1" pandas numpy datasets pyarrow
pip install -q jupyter nbconvert

echo "=== Starting training ==="
cd /workspace/app
jupyter nbconvert --to script --execute training/train_cleaning.ipynb --ExecutePreprocessor.timeout=-1

echo "=== Stage 1 Complete ==="
