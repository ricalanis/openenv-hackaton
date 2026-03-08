"""
DataSage — Stage 2: Enrichment GRPO Training (rollout_func edition)
====================================================================

Uses rollout_func + generate_rollout_completions to fix context mismatch:
the model sees real environment observations before generating actions,
and rewards evaluate against the same environment state.

Usage:
    python training/train_enrichment.py

Requires: GPU (H100 recommended), HF_TOKEN, WANDB_API_KEY env vars.
"""

import json
import os
import re
import sys
import random

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.shared.config import (
    BASE_MODEL,
    HF_MODEL_REPOS,
    SPACE_URLS,
    TRAINING_CONFIGS,
    WANDB_PROJECT,
)
from training.shared.parsers import parse_enrichment_action
from environments.enrichment.client import EnrichmentEnv
from environments.enrichment.models import EnrichmentAction

ENV_URL = SPACE_URLS["enrichment"]
STAGE_CONFIG = TRAINING_CONFIGS["enrichment"]

# ── Model loading via Unsloth ────────────────────────────────────────
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── System prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a data enrichment agent. You enrich enterprise datasets by adding \
derived fields, lookups, and computed columns across multiple domains \
(HR, Sales, Project Management, IT Operations).

Analyze the schema and available sources below, then respond with a JSON \
enrichment action:
{"operation": "<op>", "field_name": "<name>", "source": "<source>", \
"logic": "<logic>", "params": {}}

Available operations:
- add_field: Add a new enrichment field from a known source
- lookup: Look up external reference data
- compute_derived: Compute a derived metric from existing columns
- add_category: Add a categorical classification

Identify the most valuable enrichment to add and act."""

# ── Task descriptions ────────────────────────────────────────────────
TASK_DESCRIPTIONS = [
    "Enrich this dataset by adding the most valuable derived field.",
    "Add an enrichment that increases analytical coverage the most.",
    "Look at the available sources and add the most impactful one.",
    "This dataset needs enrichment. Choose the best source to add.",
    "Maximize enrichment coverage by adding the most useful field.",
    "Analyze the schema and pick the enrichment with highest value.",
    "Add a derived field that enables the most downstream analysis.",
    "Choose an enrichment source that fills the biggest analytics gap.",
]

# ── Dataset ──────────────────────────────────────────────────────────
random.seed(42)
dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": random.choice(TASK_DESCRIPTIONS)}]
        for _ in range(64)
    ]
})


# ── Prompt builder ───────────────────────────────────────────────────
def build_prompt_with_observation(obs, task_description: str) -> str:
    """Build a complete prompt with real environment observation."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Domain: {obs.domain}\n\n"
                f"Schema:\n{obs.schema_info}\n\n"
                f"Available Enrichment Sources: {', '.join(obs.available_sources)}\n\n"
                f"Possible Enrichments: {', '.join(obs.possible_enrichments)}\n\n"
                f"Data Preview:\n{obs.data_preview}\n\n"
                f"Task: {task_description}"
            )},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )


# ── Rollout function ─────────────────────────────────────────────────
def enrichment_rollout(prompts: list[str], trainer: GRPOTrainer) -> dict:
    """Custom rollout: inject env observation, evaluate against same state."""
    num_gens = trainer.args.num_generations
    tok = trainer.processing_class

    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    all_env_rewards = []
    all_coverages = []
    all_available_sources = []

    for i, _prompt in enumerate(prompts):
        seed = random.randint(0, 2**31)
        task_desc = random.choice(TASK_DESCRIPTIONS)

        # 1. Reset env with seed to get observation
        with EnrichmentEnv(base_url=ENV_URL) as client:
            reset_result = client.reset_with_seed(seed=seed)
            obs = reset_result.observation

        # 2. Build prompt with real env observation
        full_prompt = build_prompt_with_observation(obs, task_desc)

        # 3. Generate N completions for GRPO comparison
        gen_prompts = [full_prompt] * num_gens
        outputs = generate_rollout_completions(trainer, gen_prompts)

        # 4. Evaluate each completion against the same seeded env state
        for out in outputs:
            text = tok.decode(out["completion_ids"], skip_special_tokens=True)

            try:
                action_dict = parse_enrichment_action(text)
                action = EnrichmentAction(
                    operation=action_dict.get("operation", "add_field"),
                    field_name=action_dict.get("field_name", "unknown"),
                    source=action_dict.get("source", ""),
                    logic=action_dict.get("logic", ""),
                    params=action_dict.get("params", {}),
                )
                with EnrichmentEnv(base_url=ENV_URL) as client:
                    client.reset_with_seed(seed=seed, domain=obs.domain)
                    result = client.step(action)
                    env_reward = float(result.reward or 0.0)
                    coverage = result.observation.enrichment_coverage
            except Exception as e:
                print(f"Env error: {e}")
                env_reward = 0.0
                coverage = 0.0

            all_prompt_ids.append(out["prompt_ids"])
            all_completion_ids.append(out["completion_ids"])
            all_logprobs.append(out["logprobs"])
            all_env_rewards.append(env_reward)
            all_coverages.append(coverage)
            all_available_sources.append(obs.available_sources)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": all_env_rewards,
        "coverage": all_coverages,
        "available_sources": all_available_sources,
    }


# ── Reward functions ─────────────────────────────────────────────────
def env_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Primary reward from environment (passed via rollout kwargs)."""
    return [float(r) for r in kwargs.get("env_reward", [0.0] * len(completions))]


def source_relevance_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward picking a valid enrichment source for the domain."""
    sources_list = kwargs.get("available_sources", [])
    if not sources_list:
        return [0.0] * len(completions)

    rewards = []
    for text, available in zip(completions, sources_list):
        action_dict = parse_enrichment_action(text)
        source = action_dict.get("source", "")
        if source in available:
            rewards.append(1.0)
        elif action_dict.get("field_name", "unknown") != "unknown":
            rewards.append(0.3)
        else:
            rewards.append(0.0)
    return rewards


def json_format_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward well-formed JSON enrichment actions."""
    valid_ops = {"add_field", "lookup", "compute_derived", "add_category"}
    rewards = []
    for text in completions:
        if re.search(r'\{[^{}]*"operation"[^{}]*\}', text):
            try:
                match = re.search(r'\{[^{}]*"operation"[^{}]*\}', text)
                data = json.loads(match.group())
                op_ok = data.get("operation") in valid_ops
                field_ok = "field_name" in data and data["field_name"] != "unknown"
                if op_ok and field_ok:
                    rewards.append(1.0)
                elif op_ok:
                    rewards.append(0.5)
                else:
                    rewards.append(0.2)
            except (json.JSONDecodeError, AttributeError):
                rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


# ── GRPO training config ─────────────────────────────────────────────
training_args = GRPOConfig(
    output_dir="./outputs/enrichment-grpo",
    learning_rate=STAGE_CONFIG["learning_rate"],
    num_train_epochs=STAGE_CONFIG["num_train_epochs"],
    per_device_train_batch_size=STAGE_CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=STAGE_CONFIG["gradient_accumulation_steps"],
    num_generations=STAGE_CONFIG["num_generations"],
    max_completion_length=STAGE_CONFIG["max_completion_length"],
    max_prompt_length=STAGE_CONFIG["max_prompt_length"],
    logging_steps=1,
    save_steps=50,
    bf16=True,
    use_vllm=True,
    vllm_mode="colocate",
    report_to="wandb",
    run_name="datasage-enrichment-grpo-v2",
)

os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

# ── Train ─────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[
        env_reward_fn,            # Primary: environment reward (via rollout)
        source_relevance_reward,  # Relevance: valid source selection
        json_format_reward,       # Format: valid JSON output
    ],
    rollout_func=enrichment_rollout,
)

print("Starting Stage 2 (Enrichment) GRPO training v2...")
trainer.train()

# ── Save & push to Hub ───────────────────────────────────────────────
output_dir = "./outputs/enrichment-grpo-final"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Training complete! Model saved to {output_dir}")

hf_repo = HF_MODEL_REPOS["enrichment"]
print(f"Pushing to Hub: {hf_repo}")
trainer.push_to_hub(hf_repo)
print(f"Model pushed to https://huggingface.co/{hf_repo}")
