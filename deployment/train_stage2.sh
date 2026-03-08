#!/bin/bash
set -e
echo "=== DataSage Stage 2: Enrichment GRPO Training ==="
echo "GPU Info:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available (GPU may still work via CUDA)"
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch CUDA check failed"
echo ""

echo "=== Installing dependencies ==="
pip install -q unsloth trl transformers accelerate peft bitsandbytes vllm wandb python-dotenv kagglehub
pip install -q "openenv-core[core]>=0.2.1" pandas numpy datasets pyarrow

echo "=== Environment check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "=== Starting training ==="
cd /workspace/app
python training/train_enrichment.py 2>&1 | tee /workspace/train_stage2.log

echo "=== Stage 2 Complete ==="
