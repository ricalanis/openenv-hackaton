"""
OpenEnv Hackathon — GRPO Training for Data Cleaning Agent (Colab)
=================================================================
Trains an LLM to perform rigorous data cleaning using the FSDS
Medallion pipeline (Bronze → Silver), with VDS-inspired quality gates.

Statement 3.1: World Modeling — Professional Tasks
"""

# ── Cell 1 ▸ Install ─────────────────────────────────────────────────
# %%
# Install order matters: unsloth must come last to pin its own transformers version.
# vllm is intentionally NOT installed: use_vllm=False is set in GRPOConfig, and
# having any vllm version present causes trl to attempt importing GuidedDecodingParams
# at module load time, which fails on incompatible builds. Removing vllm lets
# trl's is_vllm_available() return False and skip the import entirely.
#
# !pip uninstall -y vllm                   # remove any existing vllm build
# !pip install -q "openenv-core[core]>=0.2.1"
# !pip install -q git+https://github.com/meta-pytorch/OpenEnv.git
# !pip install -q "trl>=0.12.0" "accelerate>=0.34.0" "peft>=0.13.0" "bitsandbytes>=0.43.0" "datasets>=2.20.0" "protobuf>=3.20.3,<5.0.0"
# !pip install -q unsloth                  # pins transformers; must be last
# !pip install -q "git+https://huggingface.co/spaces/israaaML/fsds_cleaning_env"


# ── Cell 2 ▸ Imports ─────────────────────────────────────────────────
# %%
import json
import re
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


# ── Cell 3 ▸ Connect to environment ──────────────────────────────────
# %%
from fsds_cleaning_env import FSDSCleaningEnv

ENV_URL = "https://israaaML-fsds-cleaning-env.hf.space"
env = FSDSCleaningEnv(base_url=ENV_URL)

# Sanity check
with env.sync() as client:
    client.reset(task_id="ecommerce_mobile")
    brief = client.call_tool("get_task_brief")
    print(f"Connected! Task: {brief.get('title')} — {brief.get('objective', '')[:200]}")
    tasks = client.call_tool("list_tasks")
    print(f"Available tasks: {[t['task_id'] for t in tasks.get('tasks', [])]}")


# ── Cell 4 ▸ Load model (Unsloth 4-bit) ─────────────────────────────
# %%
from unsloth import FastLanguageModel

# Phase 7 — SFT warm-start:
# After running training_sft.py, set MODEL_NAME to the saved SFT checkpoint
# path (e.g. "./data-cleaning-sft-final") to warm-start RL from the SFT model.
# Using an SFT checkpoint dramatically reduces the GRPO episodes needed for
# convergence because the model already knows the JSON action format and the
# correct inspect → clean → validate methodology.
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
# MODEL_NAME = "./data-cleaning-sft-final"  # ← uncomment for SFT warm-start

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=2048, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}


# ── Cell 5 ▸ System prompt & dataset ─────────────────────────────────
# %%
SYSTEM_PROMPT = """\
You are a Data Cleaning Agent working in a Medallion data pipeline (Bronze → Silver).

Your job: inspect a dirty dataset and clean it to Silver quality by choosing \
the right tools in the right order.

## Methodology (FSDS + VDS)
1. INSPECT first: profile_data, preview_data, get_task_brief
2. CLEAN systematically: fix dtypes, strip whitespace, handle missing values, \
remove duplicates, clip outliers
3. VALIDATE before submitting: run_quality_gates to check quality gate
4. SUBMIT: submit_solution when all tests pass

## Output Format
Each turn, output exactly one JSON action:
{"tool": "<tool_name>", "arguments": {"operation": "<op>", "column": "<col_or_omit>"}}

Top-level tools: profile_data, preview_data, get_task_brief, run_quality_gates, submit_solution
Cleaning tool: apply_cleaning_operation — requires an "operation" argument.

Available operations for apply_cleaning_operation:
  drop_duplicates
  replace_invalid_with_null  (requires "column")
  cast_numeric               (requires "column")
  cast_datetime              (requires "column")
  impute_numeric             (requires "column"; optional "strategy": "median"|"mean")
  impute_categorical         (requires "column")
  normalize_categories       (requires "column")
  clip_outliers_iqr          (requires "column")

Examples:
  {"tool": "profile_data", "arguments": {}}
  {"tool": "apply_cleaning_operation", "arguments": {"operation": "drop_duplicates"}}
  {"tool": "apply_cleaning_operation", "arguments": {"operation": "cast_numeric", "column": "amount"}}
  {"tool": "apply_cleaning_operation", "arguments": {"operation": "impute_numeric", "column": "amount", "strategy": "median"}}
  {"tool": "run_quality_gates", "arguments": {}}
  {"tool": "submit_solution", "arguments": {}}

Think step by step. Inspect before cleaning. Validate before submitting."""

# Multi-step prompts that simulate receiving observations mid-episode
TASK_PROMPTS = [
    "You just received a dirty Bronze-layer dataset. What is your first action?",
    "The data profile shows missing values in 'amount' (8%) and 'age' (10%), 10 duplicate rows, and mixed dtypes in 'amount'. What do you do first?",
    "You inspected the 'amount' column and found string values like '123.45 USD'. How do you fix this?",
    "After fixing dtypes, you still have 16 missing values in 'amount'. The dataset has a 'region' column. How should you impute?",
    "The 'rating' column has values outside the valid range (1-5), like 99.0 and -2.0. How do you handle this?",
    "You see whitespace issues in 'category' ('  Electronics  '). What tool do you use?",
    "The dataset has an 'email' column with PII data. What should you do?",
    "You've done all cleaning. What's the last step before submitting?",
    "Unit tests show 6/7 passed — the remaining failure is 'test_dtype_alignment' for 'rating'. What do you do?",
    "All quality gate tests pass. What's your final action?",
    "The HR dataset has inconsistent department names ('engineering', 'SALES', ' Marketing '). Fix it.",
    "Salary column has values like -5000 and 9999999. What cleaning approach do you use?",
    "The 'satisfaction' column has values like '7.5/10' mixed with floats. First fix?",
    "After cleaning, data retention is at 92%. Should you drop more rows with missing tenure, or impute?",
    "You've never profiled the data yet. A colleague tells you to just fill all NaN with 0. Is that a good approach?",
    "The quality gate shows 'test_missing_values: FAILED' with 28 remaining nulls. List your next actions.",
]


def make_conversation(user_msg: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


dataset = Dataset.from_dict({
    "prompt": [make_conversation(p) for p in TASK_PROMPTS * 4]  # 64 examples
})


# ── Cell 6 ▸ Parse model output → action ─────────────────────────────
# %%
VALID_TOOLS = {
    "list_tasks", "get_task_brief", "preview_data", "profile_data",
    "apply_cleaning_operation", "run_quality_gates", "submit_solution",
}

VALID_OPERATIONS = {
    "drop_duplicates", "replace_invalid_with_null", "cast_numeric",
    "cast_datetime", "impute_numeric", "impute_categorical",
    "normalize_categories", "clip_outliers_iqr",
}


def parse_action(text: str) -> dict:
    """Extract a tool call dict from the model's completion.

    Returns a dict with keys "tool" and "arguments" matching the
    fsds_cleaning_env call_tool() convention.
    """
    # Try JSON extraction
    json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            tool = data.get("tool", "profile_data")
            arguments = data.get("arguments", {})
            if tool in VALID_TOOLS:
                return {"tool": tool, "arguments": arguments}
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: keyword matching → build a valid tool call dict
    text_lower = text.lower()
    if "profile" in text_lower or "inspect" in text_lower:
        return {"tool": "profile_data", "arguments": {}}
    elif "duplicate" in text_lower or "dedup" in text_lower:
        return {"tool": "apply_cleaning_operation", "arguments": {"operation": "drop_duplicates"}}
    elif "invalid" in text_lower or "null" in text_lower or "replace" in text_lower:
        col_match = re.search(r"column['\"]?\s*[=:]\s*['\"]?(\w+)", text_lower)
        col = col_match.group(1) if col_match else "amount"
        return {"tool": "apply_cleaning_operation", "arguments": {"operation": "replace_invalid_with_null", "column": col}}
    elif "dtype" in text_lower or "cast" in text_lower or "numeric" in text_lower:
        col_match = re.search(r"column['\"]?\s*[=:]\s*['\"]?(\w+)", text_lower)
        col = col_match.group(1) if col_match else "amount"
        return {"tool": "apply_cleaning_operation", "arguments": {"operation": "cast_numeric", "column": col}}
    elif "impute" in text_lower or "fill" in text_lower or "missing" in text_lower:
        col_match = re.search(r"column['\"]?\s*[=:]\s*['\"]?(\w+)", text_lower)
        col = col_match.group(1) if col_match else "amount"
        return {"tool": "apply_cleaning_operation", "arguments": {"operation": "impute_numeric", "column": col, "strategy": "median"}}
    elif "clip" in text_lower or "outlier" in text_lower:
        col_match = re.search(r"column['\"]?\s*[=:]\s*['\"]?(\w+)", text_lower)
        col = col_match.group(1) if col_match else "order_value"
        return {"tool": "apply_cleaning_operation", "arguments": {"operation": "clip_outliers_iqr", "column": col}}
    elif "normaliz" in text_lower or "categor" in text_lower or "whitespace" in text_lower or "strip" in text_lower:
        col_match = re.search(r"column['\"]?\s*[=:]\s*['\"]?(\w+)", text_lower)
        col = col_match.group(1) if col_match else "device_os"
        return {"tool": "apply_cleaning_operation", "arguments": {"operation": "normalize_categories", "column": col}}
    elif "unit_test" in text_lower or "validate" in text_lower or "quality" in text_lower or "gate" in text_lower:
        return {"tool": "run_quality_gates", "arguments": {}}
    elif "submit" in text_lower:
        return {"tool": "submit_solution", "arguments": {}}
    else:
        return {"tool": "profile_data", "arguments": {}}


# ── Cell 7 ▸ Reward functions ────────────────────────────────────────
# %%
def _completion_to_text(completion) -> str:
    """TRL passes completions as chat message lists in newer versions.
    Extract the assistant's text content regardless of format.
    """
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        # fallback: last message content
        return completion[-1].get("content", "") if completion else ""
    return str(completion)


def env_reward_fn(completions, **kwargs) -> list[float]:
    """Step through the environment for each completion — PRIMARY reward."""
    rewards = []
    for completion in completions:
        text = _completion_to_text(completion)
        try:
            with FSDSCleaningEnv(base_url=ENV_URL).sync() as client:
                client.reset(task_id="ecommerce_mobile")
                tool_call = parse_action(text)
                result = client.call_tool(tool_call["tool"], **tool_call.get("arguments", {}))
                rewards.append(float(result.get("reward", 0.0)))
        except Exception as e:
            if "execution_error" not in str(e) and "Unknown column" not in str(e):
                print(f"Env error (unexpected): {e}")
            rewards.append(0.0)
    return rewards


def json_format_reward(completions, **kwargs) -> list[float]:
    """Reward valid JSON action format."""
    rewards = []
    for completion in completions:
        text = _completion_to_text(completion)
        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                tool = data.get("tool", "")
                arguments = data.get("arguments", {})
                if tool in VALID_TOOLS:
                    if tool == "apply_cleaning_operation":
                        op = arguments.get("operation", "")
                        rewards.append(1.0 if op in VALID_OPERATIONS else 0.3)
                    else:
                        rewards.append(1.0)
                else:
                    rewards.append(0.3)
            except (json.JSONDecodeError, AttributeError):
                rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


def methodology_reward(completions, **kwargs) -> list[float]:
    """
    Reward FSDS/VDS methodology:
    - Inspect before cleaning (+0.3)
    - Mention quality gate / validation (+0.2)
    - Mention PCS / stability / group-wise (+0.3)
    """
    rewards = []
    for completion in completions:
        text = _completion_to_text(completion)
        score = 0.0
        lower = text.lower()
        if any(w in lower for w in ["inspect", "profile", "first", "understand", "look at"]):
            score += 0.3
        if any(w in lower for w in ["run_quality_gates", "validate", "quality gate", "before submit"]):
            score += 0.2
        if any(w in lower for w in ["group_by", "group-wise", "per-region", "per-category", "pcs", "stability"]):
            score += 0.3
        rewards.append(min(score, 1.0))
    return rewards


# ── Cell 8 ▸ GRPO config ────────────────────────────────────────────
# %%
training_args = GRPOConfig(
    output_dir="./data-cleaning-grpo",
    learning_rate=5e-6,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=256,
    max_prompt_length=512,
    logging_steps=1,
    save_steps=50,
    fp16=True,          # T4 is Turing (not Ampere+), so bf16 is not supported
    use_vllm=False,     # vllm colocate conflicts with unsloth on T4 (16 GB)
    report_to="none",
)


# ── Cell 9 ▸ Train ──────────────────────────────────────────────────
# %%
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[
        env_reward_fn,          # Primary: environment reward
        json_format_reward,     # Format: valid JSON actions
        methodology_reward,     # Process: inspect → clean → validate
    ],
)
trainer.train()


# ── Cell 10 ▸ Save ───────────────────────────────────────────────────
# %%
trainer.save_model("./data-cleaning-grpo-final")
tokenizer.save_pretrained("./data-cleaning-grpo-final")
print("Training complete!")
