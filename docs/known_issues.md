# Known Issues

## 2026-03-08 - GRPO tensor size mismatch with Unsloth

**What happened:** `RuntimeError: The size of tensor a (254) must match the size of tensor b (255)` in GRPOTrainer.train(). Three prior fix attempts (batch_size, fp16, dtype) all failed.

**Root cause:** Missing `fast_inference=True` in `FastLanguageModel.from_pretrained()`. Without it, Unsloth falls back to HF's native `generate()` which doesn't properly pad multi-generation completions. The `use_vllm=True` in GRPOConfig requires `fast_inference=True` to use Unsloth's internal vLLM engine (not external vLLM, which is incompatible with PEFT/LoRA).

**How to avoid:** Always follow the [Unsloth GRPO reference notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb) for the exact parameter combination. Key flags: `fast_inference=True` + `use_vllm=True` are a pair — both required together.

**References:**
- [Unsloth GRPO blog](https://www.unsloth.ai/blog/grpo)
- [GRPOTrainer tensor shape error (TRL #2878)](https://github.com/huggingface/trl/issues/2878)
- [Unsloth padding_side issue (#3283)](https://github.com/unslothai/unsloth/issues/3283)
