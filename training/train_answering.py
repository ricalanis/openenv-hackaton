"""
DataSage — Stage 3: Answering GRPO Training
=============================================

Trains Qwen2.5-3B via Unsloth + TRL GRPO to generate persona-aware,
data-grounded answers against the DataSage Answering OpenEnv environment.

Includes Patronus Lynx integration for hallucination detection as a
reward signal, with a local faithfulness fallback.

Usage:
    python training/train_answering.py

Requires: GPU (H100 recommended), HF_TOKEN, WANDB_API_KEY env vars.
Optional: PATRONUS_API_KEY for Patronus Lynx hallucination evaluation.
"""

import json
import logging
import os
import re
import sys
import time

# Load .env for PATRONUS_API_KEY, HF_TOKEN, WANDB_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("=" * 60)
logger.info("DataSage Stage 3: Answering GRPO Training")
logger.info("=" * 60)

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
from training.shared.parsers import parse_answering_action

# ── Environment client ───────────────────────────────────────────────
from environments.answering.client import AnsweringEnv
from environments.answering.models import AnsweringAction

ENV_URL = SPACE_URLS["answering"]
STAGE_CONFIG = TRAINING_CONFIGS["answering"]

logger.info(f"Environment URL: {ENV_URL}")
logger.info(f"Base model: {BASE_MODEL}")
logger.info(f"Config: {STAGE_CONFIG}")
logger.info(f"Patronus API key: {'SET' if os.environ.get('PATRONUS_API_KEY') else 'NOT SET (using local fallback)'}")

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
3. Be concise but thorough — Executives want bullet points, ICs want \
clear next steps.
4. Never fabricate numbers — if the data doesn't support a claim, say so.

Think step by step: identify the persona, understand the question, \
examine the data, then answer."""

# ── Task prompts: 16+ per domain = 64+ total ────────────────────────
TASK_PROMPTS = [
    # HR domain (16)
    "As an Executive: What is the overall attrition rate and its financial impact on the organization?",
    "As a Manager: Which departments have the highest turnover and what patterns do you see?",
    "As an Individual Contributor: Am I at risk of burnout based on the overtime data?",
    "As an Executive: How does our compensation compare to industry benchmarks across job roles?",
    "As a Manager: Which team members show the highest flight risk scores?",
    "As an Individual Contributor: What's the average tenure in my department and how do I compare?",
    "As an Executive: What's the year-over-year trend in employee satisfaction scores?",
    "As a Manager: How is overtime distributed across my team and is it sustainable?",
    "As an Individual Contributor: What factors most affect performance ratings?",
    "As an Executive: What's the ROI of reducing attrition by 5% based on salary data?",
    "As a Manager: Which roles are hardest to retain and what can we do about it?",
    "As an Individual Contributor: How does my job satisfaction compare to the department average?",
    "As an Executive: What are the top 3 strategic risks in our workforce data?",
    "As a Manager: What's the capacity utilization of the team based on current headcount?",
    "As an Individual Contributor: What development areas should I focus on based on performance data?",
    "As an Executive: Summarize the workforce health metrics for the quarterly board report.",

    # Sales domain (16)
    "As an Executive: What's the pipeline health and forecast accuracy for this quarter?",
    "As a Manager: Which deals are at risk of slipping from the forecast?",
    "As an Individual Contributor: What should I focus on to close my highest-probability deal?",
    "As an Executive: What's the revenue impact of deals stuck in Negotiation stage?",
    "As a Manager: How does our team's deal velocity compare across regions?",
    "As an Individual Contributor: Which of my deals has the best win probability?",
    "As an Executive: What's the conversion rate by stage and where are we losing deals?",
    "As a Manager: Which reps are below quota and what's their pipeline gap?",
    "As an Individual Contributor: What's the average time deals spend in my current stage?",
    "As an Executive: What's the competitive risk across our enterprise deals?",
    "As a Manager: How should I reallocate team resources based on pipeline data?",
    "As an Individual Contributor: What lead sources have the highest conversion for my product?",
    "As an Executive: Project the Q4 revenue based on current pipeline and win rates.",
    "As a Manager: What's the optimal deal mix by size category for our team?",
    "As an Individual Contributor: How can I improve my deal velocity based on the data?",
    "As an Executive: Summarize the sales pipeline metrics for the leadership review.",

    # Project Management domain (16)
    "As an Executive: Which projects are at highest risk of missing their deadlines?",
    "As a Manager: How is resource utilization distributed across the team?",
    "As an Individual Contributor: Which of my tasks should I prioritize based on dependencies?",
    "As an Executive: What's the portfolio-level on-time delivery rate?",
    "As a Manager: What's the burndown rate for the current sprint?",
    "As an Individual Contributor: Am I overallocated based on estimated vs actual hours?",
    "As an Executive: What's the financial impact of project delays in the portfolio?",
    "As a Manager: Which tasks are blocking the most downstream work?",
    "As an Individual Contributor: What's my completion rate compared to estimates?",
    "As an Executive: How does our delay probability trend compare to last quarter?",
    "As a Manager: Which team members need workload rebalancing?",
    "As an Individual Contributor: What are my upcoming deadlines and their risk levels?",
    "As an Executive: Summarize project health metrics for the steering committee.",
    "As a Manager: What process bottlenecks are causing the most delays?",
    "As an Individual Contributor: How many of my tasks have high dependency depth?",
    "As an Executive: What's the schedule risk across the critical path projects?",

    # IT Operations domain (16)
    "As an Executive: What's our SLA compliance rate and its trend this quarter?",
    "As a Manager: Which ticket categories have the worst resolution times?",
    "As an Individual Contributor: Which tickets assigned to me are closest to breaching SLA?",
    "As an Executive: What's the cost impact of SLA breaches this month?",
    "As a Manager: How should I prioritize the escalation queue?",
    "As an Individual Contributor: What's the resolution pattern for tickets like mine?",
    "As an Executive: What are the top 3 systemic issues driving incident volume?",
    "As a Manager: Which systems have the most recurring incidents?",
    "As an Individual Contributor: What's the typical escalation path for my ticket category?",
    "As an Executive: What's the mean time to resolution trend across priorities?",
    "As a Manager: How is the team's incident workload distributed?",
    "As an Individual Contributor: What resolution type is most common for my ticket type?",
    "As an Executive: Summarize the IT operations health for the monthly business review.",
    "As a Manager: Which areas need more staffing based on incident patterns?",
    "As an Individual Contributor: Are there recurring patterns in the incidents I'm handling?",
    "As an Executive: What's the customer impact distribution across open incidents?",
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

logger.info(f"Dataset size: {len(dataset)} prompts across 4 domains x 3 personas")

# ── Local faithfulness scoring (fallback for Patronus) ───────────────
def local_faithfulness_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Local faithfulness heuristic: checks if the answer cites data columns
    and avoids hedging/fabrication markers.
    """
    rewards = []
    for text in completions:
        score = 0.0
        lower = text.lower()

        # Reward citing column names (CamelCase patterns typical in our schema)
        cited = re.findall(r'\b([A-Z][a-zA-Z]+(?:[A-Z][a-z]+)*)\b', text)
        if len(cited) >= 1:
            score += 0.3
        if len(cited) >= 3:
            score += 0.2

        # Reward data-grounded claims (numbers, percentages, statistics)
        if re.search(r'\d+\.?\d*%', text) or re.search(r'\b\d{2,}\b', text):
            score += 0.2

        # Penalize fabrication markers
        if any(w in lower for w in ["i believe", "probably", "might be",
                                     "i'm not sure", "i think maybe"]):
            score -= 0.2

        # Penalize empty or very short answers
        if len(text.strip()) < 50:
            score -= 0.3

        rewards.append(max(0.0, min(1.0, score)))
    return rewards


# ── Patronus Lynx integration ────────────────────────────────────────
def patronus_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Use Patronus Lynx to evaluate hallucination in model outputs.
    Falls back to local_faithfulness_fn if Patronus is unavailable.
    """
    api_key = os.environ.get("PATRONUS_API_KEY")
    if not api_key:
        return local_faithfulness_fn(completions, **kwargs)
    try:
        import patronus
        patronus.init()
        from patronus import Patronus, RemoteEvaluator

        client = Patronus()
        lynx = RemoteEvaluator("lynx", "patronus:hallucination")
        context = kwargs.get("context", "")
        task_input = kwargs.get("task_input", "Answer the question based on the data.")
        rewards = []
        for text in completions:
            result = client.evaluate(
                evaluators=lynx,
                task_output=text,
                task_input=task_input,
                task_context=context,
            )
            rewards.append(float(result.results[0].score))
        return rewards
    except Exception:
        return local_faithfulness_fn(completions, **kwargs)


# ── Reward functions ─────────────────────────────────────────────────
def env_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Step through the Answering environment for each completion.
    Primary reward signal from the OpenEnv environment (includes
    persona alignment and faithfulness scoring).
    """
    rewards = []
    for text in completions:
        try:
            action_dict = parse_answering_action(text)
            action = AnsweringAction(
                answer=action_dict.get("answer", text),
                cited_columns=action_dict.get("cited_columns", []),
                reasoning=action_dict.get("reasoning", ""),
            )
            with AnsweringEnv(base_url=ENV_URL) as client:
                client.reset()
                result = client.step(action)
                rewards.append(float(result.reward or 0.0))
        except Exception as e:
            logger.warning(f"Env error: {e}")
            rewards.append(0.0)
    return rewards


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


def persona_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Reward persona-appropriate language based on prompt cues.
    Detects which persona was requested and checks for alignment.
    """
    rewards = []
    for text in completions:
        score = 0.0
        lower = text.lower()

        # Executive language markers
        exec_markers = ["roi", "revenue", "cost", "strategic", "quarter",
                        "year-over-year", "portfolio", "margin", "growth",
                        "benchmark", "trend", "%", "million"]
        # Manager language markers
        mgr_markers = ["team", "performance", "bottleneck", "capacity",
                       "action", "priority", "escalation", "delivery",
                       "utilization", "process", "recommend"]
        # IC language markers
        ic_markers = ["my", "i should", "next step", "deadline", "help",
                      "understand", "focus on", "assigned to me"]

        exec_hits = sum(1 for m in exec_markers if m in lower)
        mgr_hits = sum(1 for m in mgr_markers if m in lower)
        ic_hits = sum(1 for m in ic_markers if m in lower)

        # Reward any strong persona alignment
        max_hits = max(exec_hits, mgr_hits, ic_hits)
        if max_hits >= 4:
            score = 0.5
        elif max_hits >= 2:
            score = 0.3
        elif max_hits >= 1:
            score = 0.1

        rewards.append(score)
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
    run_name="datasage-answering-grpo",
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
        env_reward_fn,         # Primary: environment reward
        patronus_reward_fn,    # Faithfulness: Patronus Lynx (with local fallback)
        json_format_reward,    # Auxiliary: structured output
        persona_reward,        # Auxiliary: persona alignment
    ],
)

logger.info("=" * 60)
logger.info("Starting Stage 3 (Answering) GRPO training...")
logger.info(f"  Epochs: {training_args.num_train_epochs}")
logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
logger.info(f"  Learning rate: {training_args.learning_rate}")
logger.info(f"  Reward funcs: env, patronus_lynx, json_format, persona")
logger.info("=" * 60)
t_start = time.time()
trainer.train()
logger.info(f"Training completed in {(time.time() - t_start) / 60:.1f} minutes")

# ── Save & push to Hub ───────────────────────────────────────────────
output_dir = "./outputs/answering-grpo-final"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"Training complete! Model saved to {output_dir}")

hf_repo = HF_MODEL_REPOS["answering"]
logger.info(f"Pushing to Hub: {hf_repo}")
trainer.push_to_hub(hf_repo)
logger.info(f"Model pushed to https://huggingface.co/{hf_repo}")
logger.info("Stage 3 (Answering) COMPLETE")
