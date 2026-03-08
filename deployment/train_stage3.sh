#!/bin/bash
set -e
echo "=== DataSage Stage 3: Answering GRPO Training ==="
echo "GPU Info:"
nvidia-smi
echo ""

echo "=== Installing dependencies ==="
pip install -q unsloth trl transformers accelerate peft bitsandbytes vllm wandb python-dotenv kagglehub patronus
pip install -q "openenv-core[core]>=0.2.1" pandas numpy datasets pyarrow
pip install -q jupyter nbconvert

echo "=== Starting training ==="
cd /workspace/app
jupyter nbconvert --to script --execute training/train_answering.ipynb --ExecutePreprocessor.timeout=-1

echo "=== Stage 3 Complete ==="
