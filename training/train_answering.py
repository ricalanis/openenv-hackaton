"""
DataSage — Stage 3: Answering GRPO Training (rollout_func edition)
===================================================================

Uses rollout_func + generate_rollout_completions to fix context mismatch:
the model sees real environment observations (persona, question, data)
before generating answers, and rewards evaluate against the same state.

Includes Patronus Lynx integration for hallucination detection as a
reward signal, with a local faithfulness fallback.

Usage:
    python training/train_answering.py

Requires: GPU (H100 recommended), HF_TOKEN, WANDB_API_KEY env vars.
Optional: PATRONUS_API_KEY for Patronus Lynx hallucination evaluation.
"""

import json
import os
import re
import sys
import random

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

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
from training.shared.parsers import parse_answering_action
from environments.answering.client import AnsweringEnv
from environments.answering.models import AnsweringAction
from environments.shared.personas import PERSONAS, score_persona_alignment

ENV_URL = SPACE_URLS["answering"]
STAGE_CONFIG = TRAINING_CONFIGS["answering"]

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
You are a data-driven enterprise analyst. You answer questions about \
datasets across multiple domains (HR, Sales, Project Management, \
IT Operations) tailored to the audience persona.

Personas:
- Executive: Focus on costs, ROI, strategic risk, portfolio trends, \
year-over-year metrics. Use strategic-financial language.
- Manager: Focus on team performance, operational health, process \
bottlenecks, capacity. Use operational-actionable language.
- Individual Contributor: Focus on personal tasks, deadlines, what to \
do next, simple explanations. Use plain-personal language.

Respond with a JSON answer:
{"answer": "<your answer>", "cited_columns": ["col1", "col2"], \
"reasoning": "<chain-of-thought>"}

Rules:
1. Ground every claim in the data — cite specific columns and statistics.
2. Match your tone and vocabulary to the persona.
3. Be concise but thorough.
4. Never fabricate numbers — if the data doesn't support a claim, say so."""

# ── Task descriptions ────────────────────────────────────────────────
TASK_DESCRIPTIONS = [
    "Answer the question based on the data, tailored to the persona.",
    "Provide a data-grounded answer appropriate for this audience.",
    "Analyze the data and answer the question in the persona's style.",
    "Use the dataset to answer accurately, matching the persona's focus.",
    "Generate a faithful, persona-aligned answer citing real data.",
    "Answer using statistics from the data, in the right tone for this persona.",
    "Review the data summary and answer the question for this stakeholder.",
    "Craft a response grounded in the data that matches the persona's needs.",
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
                f"Domain: {obs.domain}\n"
                f"Persona: {obs.persona}\n"
                f"Persona Focus: {obs.persona_description}\n\n"
                f"Question: {obs.question}\n\n"
                f"Dataset Summary:\n{obs.dataset_summary}\n\n"
                f"Column Statistics:\n{obs.column_stats}\n\n"
                f"Available Columns: {', '.join(obs.available_columns)}\n\n"
                f"Task: {task_description}"
            )},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )


# ── Rollout function ─────────────────────────────────────────────────
def answering_rollout(prompts: list[str], trainer: GRPOTrainer) -> dict:
    """Custom rollout: inject env observation, evaluate against same state."""
    num_gens = trainer.args.num_generations
    tok = trainer.processing_class

    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    all_env_rewards = []
    all_persona_names = []

    for i, _prompt in enumerate(prompts):
        seed = random.randint(0, 2**31)
        task_desc = random.choice(TASK_DESCRIPTIONS)

        # 1. Reset env with seed to get observation
        with AnsweringEnv(base_url=ENV_URL) as client:
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
                action_dict = parse_answering_action(text)
                action = AnsweringAction(
                    answer=action_dict.get("answer", text),
                    cited_columns=action_dict.get("cited_columns", []),
                    reasoning=action_dict.get("reasoning", ""),
                )
                with AnsweringEnv(base_url=ENV_URL) as client:
                    client.reset_with_seed(seed=seed, domain=obs.domain)
                    result = client.step(action)
                    env_reward = float(result.reward or 0.0)
            except Exception as e:
                print(f"Env error: {e}")
                env_reward = 0.0

            all_prompt_ids.append(out["prompt_ids"])
            all_completion_ids.append(out["completion_ids"])
            all_logprobs.append(out["logprobs"])
            all_env_rewards.append(env_reward)
            all_persona_names.append(obs.persona)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": all_env_rewards,
        "persona_name": all_persona_names,
    }


# ── Reward functions ─────────────────────────────────────────────────
def env_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Primary reward from environment (passed via rollout kwargs)."""
    return [float(r) for r in kwargs.get("env_reward", [0.0] * len(completions))]


def persona_match_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward alignment with the REQUESTED persona (not just any persona)."""
    persona_names = kwargs.get("persona_name", [])
    if not persona_names:
        return [0.0] * len(completions)

    persona_map = {p.name: p for p in PERSONAS}
    rewards = []
    for text, p_name in zip(completions, persona_names):
        persona = persona_map.get(p_name)
        if persona:
            rewards.append(score_persona_alignment(text, persona))
        else:
            rewards.append(0.0)
    return rewards


def local_faithfulness_fn(completions: list[str], **kwargs) -> list[float]:
    """Local faithfulness heuristic: checks column citations and data grounding."""
    rewards = []
    for text in completions:
        score = 0.0
        cited = re.findall(r'\b([A-Z][a-zA-Z]+(?:[A-Z][a-z]+)*)\b', text)
        if len(cited) >= 1:
            score += 0.3
        if len(cited) >= 3:
            score += 0.2
        if re.search(r'\d+\.?\d*%', text) or re.search(r'\b\d{2,}\b', text):
            score += 0.2
        if any(w in text.lower() for w in ["i believe", "probably", "might be",
                                             "i'm not sure", "i think maybe"]):
            score -= 0.2
        if len(text.strip()) < 50:
            score -= 0.3
        rewards.append(max(0.0, min(1.0, score)))
    return rewards


def patronus_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Use Patronus Lynx for hallucination detection, with local fallback."""
    api_key = os.environ.get("PATRONUS_API_KEY")
    if not api_key:
        return local_faithfulness_fn(completions, **kwargs)
    try:
        import patronus
        patronus.init()
        from patronus import Patronus, RemoteEvaluator
        client = Patronus()
        lynx = RemoteEvaluator("lynx", "patronus:hallucination")
        rewards = []
        for text in completions:
            result = client.evaluate(
                evaluators=lynx,
                task_output=text,
                task_input="Answer the question based on the data.",
                task_context="",
            )
            rewards.append(float(result.results[0].score))
        return rewards
    except Exception:
        return local_faithfulness_fn(completions, **kwargs)


def json_format_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward well-formed JSON answer actions."""
    rewards = []
    for text in completions:
        if re.search(r'\{[^{}]*"answer"[^{}]*\}', text, re.DOTALL):
            try:
                match = re.search(r'\{[^{}]*"answer"[^{}]*\}', text, re.DOTALL)
                data = json.loads(match.group())
                has_answer = bool(data.get("answer", "").strip())
                has_cited = bool(data.get("cited_columns"))
                has_reasoning = bool(data.get("reasoning", "").strip())
                if has_answer and has_cited and has_reasoning:
                    rewards.append(1.0)
                elif has_answer and has_cited:
                    rewards.append(0.7)
                elif has_answer:
                    rewards.append(0.4)
                else:
                    rewards.append(0.2)
            except (json.JSONDecodeError, AttributeError):
                rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


# ── GRPO training config ─────────────────────────────────────────────
training_args = GRPOConfig(
    output_dir="./outputs/answering-grpo",
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
    run_name="datasage-answering-grpo-v2",
)

os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

# ── Train ─────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[
        env_reward_fn,          # Primary: environment reward (via rollout)
        patronus_reward_fn,     # Faithfulness: Patronus Lynx (with local fallback)
        json_format_reward,     # Format: valid JSON output
        persona_match_reward,   # Persona: alignment with REQUESTED persona
    ],
    rollout_func=answering_rollout,
)

print("Starting Stage 3 (Answering) GRPO training v2...")
trainer.train()

# ── Save & push to Hub ───────────────────────────────────────────────
output_dir = "./outputs/answering-grpo-final"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Training complete! Model saved to {output_dir}")

hf_repo = HF_MODEL_REPOS["answering"]
print(f"Pushing to Hub: {hf_repo}")
trainer.push_to_hub(hf_repo)
print(f"Model pushed to https://huggingface.co/{hf_repo}")
