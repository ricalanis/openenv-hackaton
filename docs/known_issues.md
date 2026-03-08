# Known Issues

## 2026-03-08 - GRPO tensor size mismatch with Unsloth

**What happened:** `RuntimeError: The size of tensor a (254) must match the size of tensor b (255)` in GRPOTrainer.train(). Three prior fix attempts (batch_size, fp16, dtype) all failed.

**Root cause:** Missing `fast_inference=True` in `FastLanguageModel.from_pretrained()`. Without it, Unsloth falls back to HF's native `generate()` which doesn't properly pad multi-generation completions. The `use_vllm=True` in GRPOConfig requires `fast_inference=True` to use Unsloth's internal vLLM engine (not external vLLM, which is incompatible with PEFT/LoRA).

**How to avoid:** Always follow the [Unsloth GRPO reference notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb) for the exact parameter combination. Key flags: `fast_inference=True` + `use_vllm=True` are a pair — both required together.

## 2026-03-08 - GRPO completion/mask size mismatch (338 vs 256)

**What happened:** After fixing the previous tensor mismatch, a new error: `RuntimeError: The size of tensor a (338) must match the size of tensor b (256)` in `masked_batch_mean`. 256 = `max_completion_length`, 338 = actual completion length.

**Root cause:** Environment observations (DQ report + data preview + columns info + chat template) tokenize to 600-800+ tokens. `max_prompt_length=512` was too small. With vLLM, the full prompt was sent to generation. The trainer computed completion = `total_tokens - max_prompt_length`, which exceeded `max_completion_length` because the prompt was longer than 512.

**How to avoid:** Always set `max_prompt_length` large enough to cover the actual tokenized prompts. For DataSage env observations, 1024 tokens covers all domains. Verify with: `len(tokenizer.apply_chat_template(prompt, tokenize=True))` on a few examples.

**References:**
- [Unsloth GRPO blog](https://www.unsloth.ai/blog/grpo)
- [GRPOTrainer tensor shape error (TRL #2878)](https://github.com/huggingface/trl/issues/2878)
- [Unsloth padding_side issue (#3283)](https://github.com/unslothai/unsloth/issues/3283)
