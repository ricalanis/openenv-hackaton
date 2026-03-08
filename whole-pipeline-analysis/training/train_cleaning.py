"""
DataSage — Stage 1: Cleaning GRPO Training
============================================

Trains Qwen2.5-3B via Unsloth + TRL GRPO to perform multi-domain data
cleaning actions against the DataSage Cleaning OpenEnv environment.

Usage:
    python training/train_cleaning.py

Requires: GPU (H100 recommended), HF_TOKEN, WANDB_API_KEY env vars.
"""

import json
import logging
import os
import re
import sys
import time

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("=" * 60)
logger.info("DataSage Stage 1: Cleaning GRPO Training")
logger.info("=" * 60)

# ── Append project root to path so imports work ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.shared.config import (
    BASE_MODEL,
    HF_MODEL_REPOS,
    SPACE_URLS,
    TRAINING_CONFIGS,
    WANDB_PROJECT,
)
from training.shared.parsers import parse_cleaning_action

# ── Environment client ───────────────────────────────────────────────
from environments.cleaning.client import CleaningEnv
from environments.cleaning.models import CleaningAction

ENV_URL = SPACE_URLS["cleaning"]
STAGE_CONFIG = TRAINING_CONFIGS["cleaning"]

logger.info(f"Environment URL: {ENV_URL}")
logger.info(f"Base model: {BASE_MODEL}")
logger.info(f"Config: {STAGE_CONFIG}")

# ── GPU info ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    logger.warning("No GPU detected! Training will be very slow.")

# ── Model loading via Unsloth ────────────────────────────────────────
from unsloth import FastLanguageModel

logger.info("Loading model via Unsloth...")
t0 = time.time()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=16,
    gpu_memory_utilization=0.6,
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

logger.info(f"Model loaded in {time.time() - t0:.1f}s")
logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── System prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a data quality agent. You clean enterprise datasets across multiple \
domains (HR, Sales, Project Management, IT Operations).

Each turn, analyze the data and respond with a JSON cleaning action:
{"operation": "<op>", "column": "<col>", "value": "<val>", "params": {}}

Available operations:
- fill_null: Fill null values (value can be "median", "mode", or a specific value)
- fix_type: Fix type mismatches in a column (cast to proper type)
- remove_duplicate: Remove duplicate rows
- standardize: Standardize column values (strip whitespace, normalize case)
- trim: Trim whitespace from column values
- correct_typo: Correct typos in categorical values

Think step by step: examine the data quality report, identify the most \
impactful issue, then act."""

# ── Task prompts: 16+ per domain = 64+ total ────────────────────────
TASK_PROMPTS = [
    # HR domain (16)
    "This HR dataset has {n_nulls} null values in the MonthlyIncome column. Clean the data.",
    "The HR data has type mismatches in the Age column with values like 'N/A' and '#REF!'. Fix this.",
    "There are duplicate employee records in the HR dataset. Remove them to ensure data integrity.",
    "The Department column has inconsistent casing: 'sales', 'Sales', 'SALES'. Standardize it.",
    "The JobRole column has leading/trailing whitespace in several entries. Trim them.",
    "There are typos in the Attrition column: 'Yse' instead of 'Yes'. Correct the typos.",
    "The YearsAtCompany column has 23 null values and some string entries like 'unknown'. Fix the types first, then fill nulls.",
    "Multiple data quality issues exist in the HR dataset: nulls in MonthlyIncome, type errors in Age, and duplicate rows. Prioritize and fix the most impactful issue first.",
    "The DistanceFromHome column contains '#REF!' and 'TBD' values mixed with valid numbers. Clean this column.",
    "The OverTime column has entries like '  Yes  ' and 'yes' alongside proper 'Yes'/'No' values. Standardize.",
    "The Education column has null values. Fill them with the mode (most common value).",
    "The PerformanceRating column has type mismatches. Some entries are strings like 'High' instead of numeric. Fix types.",
    "The HR dataset has whitespace issues in the JobRole column. Values like ' Manager ' need trimming.",
    "Several categorical columns in the HR data have typos. Start with Department where 'Ressearch & Development' appears.",
    "The JobSatisfaction column has nulls and invalid string values. Clean this numeric column.",
    "The HR data quality score is only 0.62. Analyze the data and apply the cleaning operation that will improve quality the most.",

    # Sales domain (16)
    "The Sales data has type mismatches in the Amount column with values like 'N/A' and '#REF!'. Fix this.",
    "There are {n_nulls} null values in the Sales Amount column. Fill them with the median.",
    "The Stage column has inconsistent values: 'prospecting', 'Prospecting', 'PROSPECTING'. Standardize.",
    "Duplicate deal records exist in the sales pipeline. Remove duplicates.",
    "The Region column has whitespace issues: '  East  ', ' West'. Trim all values.",
    "The Product column has typos: 'GTX Bsaic' instead of 'GTX Basic'. Correct them.",
    "The DaysInStage column has string values like 'unknown' mixed with numbers. Fix the types.",
    "The Probability column has nulls and '#REF!' values. This is a numeric column that needs type fixing.",
    "Multiple quality issues in Sales data: Amount has nulls, Stage has inconsistencies, and there are duplicates. Fix the highest-impact issue.",
    "The ForecastCategory column values have extra whitespace. Trim them.",
    "The Sales data quality score is 0.58. The Amount column has the most issues. Clean it.",
    "The Rep column has typos in sales representative names. Correct them.",
    "The CloseDate column has invalid date formats and null values. Fix the types.",
    "The LeadSource column has inconsistent capitalization. Standardize the values.",
    "The sales pipeline has 15% duplicate records skewing forecast accuracy. Remove duplicates.",
    "The Amount column has values like '-', 'TBD', and 'N/A' alongside valid numbers. Fix type mismatches.",

    # Project Management domain (16)
    "The PM dataset has {n_nulls} nulls in EstimatedHours. Fill with the median value.",
    "The Status column has inconsistent casing: 'in progress', 'In Progress', 'IN PROGRESS'. Standardize.",
    "The ActualHours column has string values like 'N/A' and '#REF!' mixed with numbers. Fix types.",
    "Duplicate task records exist in the project data. Remove them.",
    "The Priority column has leading/trailing whitespace. Trim all values.",
    "The RiskFlag column has typos: 'Hgih' instead of 'High'. Correct them.",
    "The CompletionPct column has type mismatches. Some values are 'TBD' or 'unknown'. Fix this.",
    "Multiple issues in PM data: nulls in EstimatedHours, type errors in ActualHours, and duplicates. Fix the most impactful issue.",
    "The Assignee column has whitespace issues. Trim and standardize names.",
    "The Milestone column has inconsistent naming. Standardize the values.",
    "The PM data quality score is 0.55. The EstimatedHours column has the most nulls. Address this.",
    "The Dependencies column has type mismatches and malformed entries. Fix types.",
    "Task records have whitespace in the Status column causing grouping issues. Trim values.",
    "The ProjectName column has typos in project names. Correct them.",
    "The DueDate column has null values for 20% of tasks. Fill them appropriately.",
    "The ActualHours column has both nulls and string values like '-'. Fix types first.",

    # IT Operations domain (16)
    "The IT Ops data has type mismatches in the SLATarget column. Values like 'N/A' need fixing.",
    "There are {n_nulls} null values in the EscalationLevel column. Fill with the median.",
    "The Category column has inconsistent values: 'hardware', 'Hardware', 'HARDWARE'. Standardize.",
    "Duplicate tickets exist in the IT operations data. Remove duplicates.",
    "The Priority column has whitespace issues: '  P1-Critical  '. Trim values.",
    "The Status column has typos: 'Resovled' instead of 'Resolved'. Correct them.",
    "The ResolutionType column has inconsistent casing. Standardize all values.",
    "Multiple quality issues in IT Ops: SLATarget has type errors, Category is inconsistent, and there are duplicates. Prioritize.",
    "The CustomerImpact column has leading/trailing spaces. Trim all values.",
    "The AffectedSystem column has typos in system names. Correct them.",
    "The IT Ops data quality score is 0.60. Identify and fix the most impactful issue.",
    "The EscalationLevel column has string values like 'unknown' mixed with numbers. Fix types.",
    "The CreatedDate column has null values and invalid formats. Address this.",
    "The Assignee column has whitespace and casing issues. Standardize.",
    "Ticket records have duplicate entries causing inflated incident counts. Remove duplicates.",
    "The SLATarget column has both nulls and '#REF!' values. Fix types and fill nulls.",
]


def make_conversation(user_msg: str) -> list[dict]:
    """Build a chat-format prompt for GRPO."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# Fill placeholders with realistic values
import random
random.seed(42)

filled_prompts = []
for p in TASK_PROMPTS:
    filled = p.format(n_nulls=random.randint(5, 45))
    filled_prompts.append(filled)

dataset = Dataset.from_dict({
    "prompt": [make_conversation(p) for p in filled_prompts]
})

logger.info(f"Dataset size: {len(dataset)} prompts across 4 domains")

# ── Reward functions ─────────────────────────────────────────────────
_reward_call_count = {"env": 0, "json": 0, "reasoning": 0}
def env_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Step through the Cleaning environment for each completion.
    Primary reward signal from the OpenEnv environment.
    """
    rewards = []
    _reward_call_count["env"] += 1
    for text in completions:
        try:
            action_dict = parse_cleaning_action(text)
            action = CleaningAction(
                operation=action_dict.get("operation", "fill_null"),
                column=action_dict.get("column", ""),
                value=action_dict.get("value"),
                params=action_dict.get("params", {}),
            )
            with CleaningEnv(base_url=ENV_URL) as client:
                client.reset()
                result = client.step(action)
                rewards.append(float(result.reward or 0.0))
        except Exception as e:
            logger.warning(f"Env error: {e}")
            rewards.append(0.0)
    if _reward_call_count["env"] % 10 == 1:
        logger.info(f"[env_reward] call #{_reward_call_count['env']}, batch avg={sum(rewards)/len(rewards):.3f}")
    return rewards


def json_format_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward well-formed JSON cleaning actions."""
    rewards = []
    for text in completions:
        if re.search(r'\{[^{}]*"operation"[^{}]*\}', text):
            try:
                match = re.search(r'\{[^{}]*"operation"[^{}]*\}', text)
                data = json.loads(match.group())
                valid_ops = {"fill_null", "fix_type", "remove_duplicate",
                             "standardize", "trim", "correct_typo"}
                if data.get("operation") in valid_ops and "column" in data:
                    rewards.append(1.0)
                elif data.get("operation") in valid_ops:
                    rewards.append(0.6)
                else:
                    rewards.append(0.3)
            except (json.JSONDecodeError, AttributeError):
                rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


def reasoning_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward chain-of-thought reasoning before the action."""
    rewards = []
    for text in completions:
        score = 0.0
        lower = text.lower()
        # Reward reasoning indicators
        if any(w in lower for w in ["first", "let me", "i should", "step 1",
                                     "think", "analyze", "examining"]):
            score += 0.3
        # Reward mentioning specific DQ issues
        if any(w in lower for w in ["null", "missing", "type", "duplicate",
                                     "whitespace", "typo", "inconsistent"]):
            score += 0.2
        # Reward mentioning column names
        if re.search(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', text):
            score += 0.1
        rewards.append(min(score, 0.5))
    return rewards


# ── GRPO training config ─────────────────────────────────────────────
training_args = GRPOConfig(
    output_dir="./outputs/cleaning-grpo",
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
    run_name="datasage-cleaning-grpo",
)

# Set W&B project
os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

# ── Train ─────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[
        env_reward_fn,       # Primary: environment reward
        json_format_reward,  # Auxiliary: structured output
        reasoning_reward,    # Auxiliary: chain-of-thought
    ],
)

logger.info("=" * 60)
logger.info("Starting Stage 1 (Cleaning) GRPO training...")
logger.info(f"  Epochs: {training_args.num_train_epochs}")
logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
logger.info(f"  Grad accum: {training_args.gradient_accumulation_steps}")
logger.info(f"  Num generations: {training_args.num_generations}")
logger.info(f"  Learning rate: {training_args.learning_rate}")
logger.info("=" * 60)
t_start = time.time()
trainer.train()
logger.info(f"Training completed in {(time.time() - t_start) / 60:.1f} minutes")

# ── Save & push to Hub ───────────────────────────────────────────────
output_dir = "./outputs/cleaning-grpo-final"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"Model saved to {output_dir}")

hf_repo = HF_MODEL_REPOS["cleaning"]
logger.info(f"Pushing to Hub: {hf_repo}")
trainer.push_to_hub(hf_repo)
logger.info(f"Model pushed to https://huggingface.co/{hf_repo}")
logger.info("=" * 60)
logger.info("Stage 1 (Cleaning) COMPLETE")
logger.info("=" * 60)
