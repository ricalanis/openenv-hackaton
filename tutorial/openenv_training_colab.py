"""
OpenEnv Hackathon — GRPO Training Script (Colab-ready)
======================================================
Personal Assistant World Modeling Environment (Statement 3.2)

Trains a small LLM via GRPO to handle scheduling conflicts, email
triage, and task delegation using OpenEnv + Unsloth/TRL.

Paste each "# ── Cell N" block into a separate Colab cell.
"""

# ── Cell 1 ▸ Install dependencies ────────────────────────────────────
# %%
# !pip install -q "openenv-core[core]>=0.2.1"
# !pip install -q git+https://github.com/meta-pytorch/OpenEnv.git
# !pip install -q trl>=0.26 transformers accelerate peft bitsandbytes
# !pip install -q vllm
# !pip install -q unsloth   # Comment out for plain HF path
#
# Install YOUR environment client from your HF Space:
# !pip install -q "git+https://huggingface.co/spaces/YOUR_USERNAME/my-env"


# ── Cell 2 ▸ Imports ────────────────────────────────────────────────
# %%
import json
import re
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


# ── Cell 3 ▸ Connect to your OpenEnv environment ────────────────────
# %%
from my_env.client import MyEnv
from my_env.models import MyAction

# Point to YOUR deployed HF Space
ENV_URL = "https://YOUR_USERNAME-my-env.hf.space"  # ← REPLACE
env = MyEnv(base_url=ENV_URL)

# Quick sanity check
with env.sync() as client:
    obs, reward = client.reset()
    print("Environment connected!")
    print(f"Task: {obs.result[:100]}...")
    print(f"Tools: {obs.available_tools}")


# ── Cell 4 ▸ Load model (Unsloth 4-bit — fits free T4) ──────────────
# %%
USE_UNSLOTH = True  # Set False for plain HF + PEFT

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"

if USE_UNSLOTH:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
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
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    peft_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ── Cell 5 ▸ System prompt & dataset ────────────────────────────────
# %%
SYSTEM_PROMPT = """\
You are a personal assistant AI. You handle scheduling conflicts, \
email triage, and task management.

Each turn, respond with a JSON action:
{"tool_name": "<tool>", "tool_args": {"key": "value"}}

Available tools: check_calendar, check_inbox, send_email, \
send_message, reschedule_meeting, delegate_task

For send_email: use tool_args {"to": "...", "subject": "...", "body": "..."}
For send_message: use tool_args {"to": "...", "body": "..."}
For reschedule_meeting: use tool_args {"new_time": "..."}
For delegate_task: use tool_args {"to": "...", "task": "..."}

Think step by step: first gather information, then act decisively."""

# Prompt variations to train on
TASK_PROMPTS = [
    "You have a dinner at 7pm but a mandatory meeting was just scheduled at 6:30pm. Handle this conflict.",
    "Your inbox has 5 emails including an urgent client escalation. Triage and respond appropriately.",
    "You need to reschedule a 1:1 with your manager because of a doctor appointment. Handle it professionally.",
    "A colleague asked you to cover their presentation tomorrow but you have a conflicting deadline. Resolve this.",
    "Your partner texted that dinner plans changed to 8pm but you have a late meeting. Figure it out.",
    "You got an urgent Slack from the CEO while on PTO. Decide how to handle it.",
    "Three meetings overlap tomorrow between 2-4pm. Prioritize and reschedule as needed.",
    "A client email needs a response within the hour but you're in back-to-back meetings. Delegate or respond.",
]


def make_conversation(user_msg):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


dataset = Dataset.from_dict({
    "prompt": [make_conversation(p) for p in TASK_PROMPTS * 8]  # 64 examples
})


# ── Cell 6 ▸ Parse model output into actions ────────────────────────
# %%
def parse_action(text: str) -> MyAction:
    """Try to extract a JSON action from the model's completion."""
    # Try to find JSON in the text
    json_match = re.search(r'\{[^{}]*"tool_name"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return MyAction(**data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: look for tool name keywords
    text_lower = text.lower()
    if "check_calendar" in text_lower:
        return MyAction(tool_name="check_calendar")
    elif "check_inbox" in text_lower:
        return MyAction(tool_name="check_inbox")
    elif "reschedule" in text_lower:
        return MyAction(tool_name="reschedule_meeting",
                        tool_args={"new_time": "earlier"})
    elif "send_email" in text_lower or "email" in text_lower:
        return MyAction(tool_name="send_email",
                        tool_args={"to": "boss@work.com",
                                   "subject": "Schedule",
                                   "body": "reschedule meeting"})
    elif "send_message" in text_lower or "message" in text_lower:
        return MyAction(tool_name="send_message",
                        tool_args={"to": "partner",
                                   "body": "running late, sorry"})
    elif "delegate" in text_lower:
        return MyAction(tool_name="delegate_task",
                        tool_args={"to": "teammate", "task": "handle this"})

    # If nothing matches, default to gathering info
    return MyAction(tool_name="check_calendar")


# ── Cell 7 ▸ Reward functions ────────────────────────────────────────
# %%
def env_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Step through the environment for each completion.
    This is the PRIMARY reward signal.
    """
    rewards = []
    for text in completions:
        try:
            with env.sync() as client:
                client.reset()
                action = parse_action(text)
                result = client.step(action)
                rewards.append(float(result.reward))
        except Exception as e:
            print(f"Env error: {e}")
            rewards.append(0.0)
    return rewards


def json_format_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward well-formed JSON actions — encourages structured output."""
    rewards = []
    for text in completions:
        if re.search(r'\{[^{}]*"tool_name"[^{}]*\}', text):
            try:
                match = re.search(r'\{[^{}]*"tool_name"[^{}]*\}', text)
                json.loads(match.group())
                rewards.append(1.0)   # valid JSON with tool_name
            except (json.JSONDecodeError, AttributeError):
                rewards.append(0.2)   # has the pattern but invalid
        else:
            rewards.append(0.0)
    return rewards


def reasoning_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward chain-of-thought before acting."""
    rewards = []
    for text in completions:
        score = 0.0
        lower = text.lower()
        # Reward if model reasons before giving an action
        if any(w in lower for w in ["first", "let me", "i should", "step 1", "think"]):
            score += 0.3
        # Reward mentioning the conflict
        if any(w in lower for w in ["conflict", "overlap", "clash", "problem"]):
            score += 0.2
        rewards.append(score)
    return rewards


# ── Cell 8 ▸ GRPO config ────────────────────────────────────────────
# %%
training_args = GRPOConfig(
    output_dir="./openenv-grpo-output",
    learning_rate=5e-6,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=256,
    max_prompt_length=512,
    logging_steps=1,
    save_steps=50,
    bf16=True,
    use_vllm=True,
    vllm_mode="colocate",
    report_to="none",  # Change to "wandb" for experiment tracking
)


# ── Cell 9 ▸ Train! ─────────────────────────────────────────────────
# %%
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[
        env_reward_fn,         # Primary: environment reward
        json_format_reward,    # Auxiliary: structured output
        reasoning_reward,      # Auxiliary: chain-of-thought
    ],
)

trainer.train()


# ── Cell 10 ▸ Save & (optionally) push to Hub ───────────────────────
# %%
trainer.save_model("./openenv-grpo-final")
tokenizer.save_pretrained("./openenv-grpo-final")
print("Training complete! Model saved to ./openenv-grpo-final")

# Uncomment to push to Hub:
# from huggingface_hub import login
# login()
# trainer.push_to_hub("YOUR_USERNAME/personal-assistant-grpo")
