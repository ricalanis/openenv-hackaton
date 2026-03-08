#!/bin/bash
set -e
echo "=== DataSage Stage 3: Answering GRPO Training ==="
echo "GPU Info:"
nvidia-smi
echo ""

echo "=== Installing dependencies ==="
pip install -q unsloth trl transformers accelerate peft bitsandbytes vllm wandb python-dotenv kagglehub patronus
pip install -q "openenv-core[core]>=0.2.1" pandas numpy datasets pyarrow

echo "=== Environment check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "=== Starting training ==="
cd /workspace/app
python training/train_answering.py 2>&1 | tee /workspace/train_stage3.log

echo "=== Stage 3 Complete ==="
