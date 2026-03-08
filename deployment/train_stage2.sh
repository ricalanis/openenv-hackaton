#!/bin/bash
set -e
echo "=== DataSage Stage 2: Enrichment GRPO Training ==="
echo "GPU Info:"
nvidia-smi
echo ""

echo "=== Installing dependencies ==="
pip install -q unsloth trl transformers accelerate peft bitsandbytes vllm wandb python-dotenv kagglehub
pip install -q "openenv-core[core]>=0.2.1" pandas numpy datasets pyarrow

echo "=== Starting training ==="
cd /workspace/app
python training/train_enrichment.py

echo "=== Stage 2 Complete ==="
