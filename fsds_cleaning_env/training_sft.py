"""
SFT Training for the FSDS Cleaning Agent (Colab)
=================================================
Phase 7 — SFT-first, RL-second

Trains an LLM via Supervised Fine-Tuning (SFT) on expert demonstrations
collected from the HeuristicAgent.  The resulting checkpoint is the warm-start
for subsequent GRPO reinforcement learning (see training_colab.py).

Pipeline
--------
1. Collect demonstrations from the HeuristicAgent (or load from disk).
2. Format them as step-level SFT examples: each step is a (prompt, completion)
   pair where the completion is the JSON action the model should predict.
3. Fine-tune with trl.SFTTrainer using Unsloth for memory efficiency.
4. Save the SFT adapter → use MODEL_NAME in training_colab.py for RL warm-start.
"""

# ── Cell 1 ▸ Install ──────────────────────────────────────────────────────────
# %%
# !pip uninstall -y vllm
# !pip install -q "openenv-core[core]>=0.2.1"
# !pip install -q "trl>=0.12.0" "accelerate>=0.34.0" "peft>=0.13.0" "bitsandbytes>=0.43.0" "datasets>=2.20.0"
# !pip install -q unsloth
# !pip install -q "git+https://huggingface.co/spaces/israaaML/fsds_cleaning_env"


# ── Cell 2 ▸ Configuration ────────────────────────────────────────────────────
# %%
import json
from pathlib import Path

# ---- Paths ------------------------------------------------------------------
DEMO_PATH       = "./demos/expert_demos.json"   # where demonstrations are saved/loaded
SFT_OUTPUT_DIR  = "./data-cleaning-sft"         # SFT adapter checkpoint
SFT_FINAL_DIR   = "./data-cleaning-sft-final"   # merged/saved final model

# ---- Environment ------------------------------------------------------------
ENV_URL = "https://israaaML-fsds-cleaning-env.hf.space"
# ENV_URL = "http://localhost:8000"  # local server

# ---- Model ------------------------------------------------------------------
BASE_MODEL      = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH  = 2048

# ---- Data collection --------------------------------------------------------
COLLECT_FRESH   = True      # set False to load from DEMO_PATH
N_PER_TASK      = 20        # demonstrations per task (3 tasks → 60 total)
SFT_MODE        = "step"    # "step" (one example per action) | "episode" (multi-turn)


# ── Cell 3 ▸ Collect / load demonstrations ───────────────────────────────────
# %%
from fsds_cleaning_env.demonstrations import (
    DemonstrationCollector,
    build_sft_dataset,
    demo_stats,
    load_demonstrations,
    save_demonstrations,
)
from fsds_cleaning_env import FSDSCleaningEnv

if COLLECT_FRESH:
    print("Collecting demonstrations from HeuristicAgent …")
    with FSDSCleaningEnv(base_url=ENV_URL).sync() as env:
        collector = DemonstrationCollector(env)
        demos = collector.collect(
            task_ids=["ecommerce_mobile", "subscription_churn", "delivery_eta"],
            n_per_task=N_PER_TASK,
            seed_offset=1000,       # use a held-out seed range
        )
    save_demonstrations(demos, DEMO_PATH)
else:
    demos = load_demonstrations(DEMO_PATH)

print("\nDemonstration statistics:")
print(json.dumps(demo_stats(demos), indent=2))


# ── Cell 4 ▸ Build SFT dataset ───────────────────────────────────────────────
# %%
dataset = build_sft_dataset(demos, mode=SFT_MODE, successful_only=True)
print(f"\nSFT dataset: {len(dataset)} examples")
print("Sample:", json.dumps(dataset[0], indent=2, ensure_ascii=False)[:400], "…")


# ── Cell 5 ▸ Load model (Unsloth 4-bit) ─────────────────────────────────────
# %%
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ── Cell 6 ▸ Format dataset for SFTTrainer ───────────────────────────────────
# %%
def _format_step_example(example: dict) -> dict:
    """Convert a step-level example to a chat-formatted string for SFTTrainer."""
    messages = example["prompt"] + [
        {"role": "assistant", "content": example["completion"]}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def _format_episode_example(example: dict) -> dict:
    """Convert an episode-level multi-turn example to a formatted string."""
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


if SFT_MODE == "step":
    formatted = dataset.map(_format_step_example, remove_columns=dataset.column_names)
else:
    formatted = dataset.map(_format_episode_example, remove_columns=dataset.column_names)

print(f"Formatted {len(formatted)} SFT examples.")
print("First example text (truncated):", formatted[0]["text"][:300], "…")


# ── Cell 7 ▸ SFT training config ─────────────────────────────────────────────
# %%
from trl import SFTConfig, SFTTrainer

sft_args = SFTConfig(
    output_dir=SFT_OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=5,
    save_steps=100,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    fp16=True,          # T4 GPU (Turing) — use fp16, not bf16
    report_to="none",
    dataset_num_proc=2,
)


# ── Cell 8 ▸ Train ───────────────────────────────────────────────────────────
# %%
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_args,
    train_dataset=formatted,
)

print("Starting SFT training …")
trainer.train()


# ── Cell 9 ▸ Save ────────────────────────────────────────────────────────────
# %%
trainer.save_model(SFT_FINAL_DIR)
tokenizer.save_pretrained(SFT_FINAL_DIR)
print(f"\nSFT checkpoint saved to: {SFT_FINAL_DIR}")
print(
    "\nNext step — RL fine-tuning:\n"
    "  Open training_colab.py and set:\n"
    f"    MODEL_NAME = '{SFT_FINAL_DIR}'\n"
    "  Then run the GRPO training cells to continue with RL.\n"
    "  The SFT checkpoint provides a warm-start that dramatically\n"
    "  reduces the number of RL episodes needed for convergence."
)
