"""
DataSage — Stage 2: Enrichment GRPO Training
==============================================

Trains Qwen2.5-3B via Unsloth + TRL GRPO to perform multi-domain data
enrichment actions against the DataSage Enrichment OpenEnv environment.

Usage:
    python training/train_enrichment.py

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

# ── Append project root to path so imports work ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.shared.config import (
    BASE_MODEL,
    HF_MODEL_REPOS,
    SPACE_URLS,
    TRAINING_CONFIGS,
    WANDB_PROJECT,
)
from training.shared.parsers import parse_enrichment_action

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("=" * 60)
logger.info("DataSage Stage 2: Enrichment GRPO Training")
logger.info("=" * 60)

# ── Environment client ───────────────────────────────────────────────
from environments.enrichment.client import EnrichmentEnv
from environments.enrichment.models import EnrichmentAction

ENV_URL = SPACE_URLS["enrichment"]
STAGE_CONFIG = TRAINING_CONFIGS["enrichment"]

logger.info(f"Environment URL: {ENV_URL}")
logger.info(f"Base model: {BASE_MODEL}")
logger.info(f"Config: {STAGE_CONFIG}")

# ── Model loading via Unsloth ────────────────────────────────────────
from unsloth import FastLanguageModel

if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
logger.info("Loading model via Unsloth...")
t0 = time.time()

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

logger.info(f"Model loaded in {time.time() - t0:.1f}s")
logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── System prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a data enrichment agent. You enrich enterprise datasets by adding \
derived fields, lookups, and computed columns across multiple domains \
(HR, Sales, Project Management, IT Operations).

Each turn, analyze the schema and respond with a JSON enrichment action:
{"operation": "<op>", "field_name": "<name>", "source": "<source>", \
"logic": "<logic>", "params": {}}

Available operations:
- add_field: Add a new enrichment field from a known source
- lookup: Look up external reference data
- compute_derived: Compute a derived metric from existing columns
- add_category: Add a categorical classification

Available enrichment sources per domain:
- HR: salary_band, tenure_risk, satisfaction_index, industry_benchmark, \
flight_risk_score
- Sales: deal_size_category, velocity_score, win_probability_model, \
industry_code, competitive_risk
- PM: schedule_risk_score, resource_utilization, dependency_chain_depth, \
burndown_rate, delay_probability
- IT Ops: sla_compliance_flag, mttr_band, escalation_path, \
incident_severity_score, recurring_pattern_flag

Think step by step: examine the current schema, identify what enrichments \
would add the most value, then act."""

# ── Task prompts: 16+ per domain = 64+ total ────────────────────────
TASK_PROMPTS = [
    # HR domain (16)
    "This HR dataset has employee data with MonthlyIncome but no salary band classification. Add a salary_band enrichment.",
    "The HR data has YearsAtCompany but no risk assessment. Add tenure_risk to identify flight risk by tenure.",
    "We need a satisfaction_index that combines multiple satisfaction factors. Enrich the HR data.",
    "Add industry_benchmark salary data so we can compare our compensation to market rates.",
    "Compute a flight_risk_score combining tenure, satisfaction, and overtime factors for each employee.",
    "The HR dataset needs salary bands for compensation analysis. Which enrichment should be added first?",
    "We want to identify employees at risk of leaving. Add the most relevant enrichment fields.",
    "Enrich this HR data with all available sources to maximize coverage. Start with the most impactful.",
    "The dataset has JobRole and MonthlyIncome. Add salary_band to categorize compensation levels.",
    "Management wants to understand attrition drivers. Add tenure_risk and satisfaction_index.",
    "The HR data enrichment coverage is only 20%. Add fields to improve analytical value.",
    "We need to benchmark our salaries against the industry. Add the industry_benchmark field.",
    "Add a computed flight risk metric that considers multiple employee factors.",
    "The HR schema is missing derived analytics. Add satisfaction_index as a composite metric.",
    "Enrich employee records with salary band classifications for the compensation review.",
    "The HR data has raw metrics but no risk scores. Start enriching with tenure_risk.",

    # Sales domain (16)
    "The Sales pipeline has Amount data but no deal categorization. Add deal_size_category.",
    "We need to track deal velocity. Add velocity_score based on DaysInStage benchmarks.",
    "Add a win_probability_model to predict deal outcomes from stage and velocity data.",
    "The sales data needs industry classification codes. Add industry_code from account patterns.",
    "Compute competitive_risk scores to identify deals at risk of being lost to competitors.",
    "The sales pipeline lacks predictive metrics. Add win_probability_model as the first enrichment.",
    "Enrich deals with size categories and velocity scores for pipeline analysis.",
    "The Sales data enrichment coverage is 0%. Start adding the most valuable enrichment fields.",
    "We need to categorize deals for forecasting. Add deal_size_category based on Amount.",
    "Sales leadership wants velocity metrics. Add velocity_score to track deal progression speed.",
    "Add industry codes to the sales data for market segmentation analysis.",
    "The pipeline needs risk scoring. Add competitive_risk based on deal stage and velocity.",
    "Enrich the sales dataset to support better forecasting. Start with win_probability_model.",
    "Add deal size categorization so we can segment the pipeline by Small/Medium/Large/Enterprise.",
    "The sales team needs velocity analytics. Enrich with velocity_score from DaysInStage.",
    "Add all available sales enrichments to maximize analytical coverage. Start with the most impactful.",

    # Project Management domain (16)
    "The PM dataset has CompletionPct but no schedule risk assessment. Add schedule_risk_score.",
    "We need resource utilization metrics. Add resource_utilization from EstimatedHours and ActualHours.",
    "Add dependency_chain_depth to understand task blocking relationships.",
    "Compute burndown_rate to track project completion velocity against plan.",
    "Add delay_probability to predict which tasks are likely to miss their deadlines.",
    "The PM data needs risk scoring. Add schedule_risk_score as the highest-priority enrichment.",
    "Enrich project tasks with utilization and burndown metrics for sprint analysis.",
    "The PM enrichment coverage is very low. Add fields to improve project analytics.",
    "We need to identify at-risk tasks. Add schedule_risk_score and delay_probability.",
    "Add resource utilization tracking to understand team capacity.",
    "The project data has completion percentages but no rate metrics. Add burndown_rate.",
    "Enrich tasks with dependency depth to identify critical path bottlenecks.",
    "Project managers need delay predictions. Add delay_probability from current trajectories.",
    "Add schedule risk scoring to the PM dataset for deadline management.",
    "The PM data has hours data but no utilization ratio. Add resource_utilization.",
    "Enrich project tasks with all available PM enrichments. Start with schedule_risk_score.",

    # IT Operations domain (16)
    "The IT Ops data has SLA targets but no compliance flags. Add sla_compliance_flag.",
    "We need MTTR classification. Add mttr_band to categorize resolution speed.",
    "Add escalation_path recommendations based on ticket category and priority.",
    "Compute incident_severity_score from priority and customer impact factors.",
    "Add recurring_pattern_flag to identify tickets that are part of recurring issues.",
    "The IT Ops data needs SLA monitoring. Add sla_compliance_flag as the first enrichment.",
    "Enrich tickets with severity scoring and MTTR bands for operations dashboards.",
    "The IT Ops enrichment coverage is 0%. Start adding enrichments for incident analysis.",
    "We need to classify resolution times. Add mttr_band from escalation data.",
    "Add escalation path recommendations for the incident management workflow.",
    "The ticket data needs a severity score. Add incident_severity_score from priority and impact.",
    "Add recurring pattern detection to identify systemic issues in IT operations.",
    "Enrich IT tickets with SLA compliance flags for the monthly compliance report.",
    "The ops team needs MTTR analytics. Add mttr_band to classify resolution speed.",
    "Add all available IT Ops enrichments for comprehensive incident analytics.",
    "We need to flag recurring incidents. Add recurring_pattern_flag to the ticket data.",
]


def make_conversation(user_msg: str) -> list[dict]:
    """Build a chat-format prompt for GRPO."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


dataset = Dataset.from_dict({
    "prompt": [make_conversation(p) for p in TASK_PROMPTS]
})

logger.info(f"Dataset size: {len(dataset)} prompts across 4 domains")

# ── Reward functions ─────────────────────────────────────────────────
def env_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Step through the Enrichment environment for each completion.
    Primary reward signal from the OpenEnv environment.
    """
    rewards = []
    for text in completions:
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
                client.reset()
                result = client.step(action)
                rewards.append(float(result.reward or 0.0))
        except Exception as e:
            logger.warning(f"Env error: {e}")
            rewards.append(0.0)
    return rewards


def json_format_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward well-formed JSON enrichment actions."""
    rewards = []
    valid_ops = {"add_field", "lookup", "compute_derived", "add_category"}
    valid_sources = {
        "salary_band", "tenure_risk", "satisfaction_index", "industry_benchmark",
        "flight_risk_score", "deal_size_category", "velocity_score",
        "win_probability_model", "industry_code", "competitive_risk",
        "schedule_risk_score", "resource_utilization", "dependency_chain_depth",
        "burndown_rate", "delay_probability", "sla_compliance_flag", "mttr_band",
        "escalation_path", "incident_severity_score", "recurring_pattern_flag",
    }
    for text in completions:
        if re.search(r'\{[^{}]*"operation"[^{}]*\}', text):
            try:
                match = re.search(r'\{[^{}]*"operation"[^{}]*\}', text)
                data = json.loads(match.group())
                op_ok = data.get("operation") in valid_ops
                field_ok = "field_name" in data and data["field_name"] != "unknown"
                source_ok = data.get("source", "") in valid_sources
                if op_ok and field_ok and source_ok:
                    rewards.append(1.0)
                elif op_ok and field_ok:
                    rewards.append(0.6)
                elif op_ok:
                    rewards.append(0.3)
                else:
                    rewards.append(0.2)
            except (json.JSONDecodeError, AttributeError):
                rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


def reasoning_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward chain-of-thought reasoning before the enrichment action."""
    rewards = []
    for text in completions:
        score = 0.0
        lower = text.lower()
        # Reward reasoning indicators
        if any(w in lower for w in ["first", "let me", "i should", "step 1",
                                     "think", "analyze", "examining", "looking at"]):
            score += 0.3
        # Reward mentioning enrichment rationale
        if any(w in lower for w in ["enrich", "add", "derive", "compute",
                                     "coverage", "value", "analytical"]):
            score += 0.2
        rewards.append(min(score, 0.5))
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
    run_name="datasage-enrichment-grpo",
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
logger.info("Starting Stage 2 (Enrichment) GRPO training...")
logger.info(f"  Epochs: {training_args.num_train_epochs}")
logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
logger.info(f"  Learning rate: {training_args.learning_rate}")
logger.info("=" * 60)
t_start = time.time()

trainer.train()

logger.info(f"Training completed in {(time.time() - t_start) / 60:.1f} minutes")

# ── Save & push to Hub ───────────────────────────────────────────────
output_dir = "./outputs/enrichment-grpo-final"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"Training complete! Model saved to {output_dir}")

hf_repo = HF_MODEL_REPOS["enrichment"]
logger.info(f"Pushing to Hub: {hf_repo}")
trainer.push_to_hub(hf_repo)
logger.info(f"Model pushed to https://huggingface.co/{hf_repo}")

logger.info("Stage 2 (Enrichment) COMPLETE")
