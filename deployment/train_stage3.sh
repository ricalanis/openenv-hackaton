#!/bin/bash
set -e
echo "=== DataSage Stage 3: Answering GRPO Training ==="
echo "GPU Info:"
nvidia-smi
echo ""

echo "=== Installing dependencies ==="
pip install -q unsloth trl transformers accelerate peft bitsandbytes vllm wandb python-dotenv kagglehub patronus
pip install -q "openenv-core[core]>=0.2.1" pandas numpy datasets pyarrow

echo "=== Starting training ==="
cd /workspace/app
python training/train_answering.py

echo "=== Stage 3 Complete ==="
