#!/usr/bin/env python3
"""
Real benchmark runner for DataSage demo.
Runs GPT-4o-mini and Qwen3-8B against live HF Space environments.
Collects actual metrics and saves results.
"""

import json
import os
import re
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

SPACE_URLS = {
    "cleaning": "https://ricalanis-datasage-cleaning.hf.space",
    "enrichment": "https://ricalanis-datasage-enrichment.hf.space",
    "answering": "https://ricalanis-datasage-answering.hf.space",
}

# Models to benchmark
MODELS_CONFIG = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "display_name": "GPT-4o-mini",
    },
    "qwen3-8b": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3-8B",
        "hf_provider": "fireworks-ai",
        "display_name": "Qwen3-8B",
    },
}

N_EPISODES = 3  # episodes per task/domain combo

# ---------------------------------------------------------------------------
# Model providers
# ---------------------------------------------------------------------------

def call_openai(prompt: str, system_prompt: str = "", max_tokens: int = 300) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, max_tokens=max_tokens, temperature=0.7,
    )
    return resp.choices[0].message.content


def call_qwen(prompt: str, system_prompt: str = "", max_tokens: int = 300) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(provider="fireworks-ai", token=HF_TOKEN or None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat_completion(
        model="Qwen/Qwen3-8B", messages=messages, max_tokens=max_tokens, temperature=0.7,
    )
    content = resp.choices[0].message.content
    # Qwen3 uses <think>...</think> tags - strip thinking
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content


def call_model(model_key: str, prompt: str, system_prompt: str = "", max_tokens: int = 300) -> str:
    if model_key == "gpt-4o-mini":
        return call_openai(prompt, system_prompt, max_tokens)
    elif model_key == "qwen3-8b":
        return call_qwen(prompt, system_prompt, max_tokens)
    raise ValueError(f"Unknown model: {model_key}")


# ---------------------------------------------------------------------------
# Environment interaction
# ---------------------------------------------------------------------------
import requests


def env_reset(task: str) -> dict:
    url = f"{SPACE_URLS[task]}/web/reset"
    r = requests.post(url, json={}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(task: str, action: dict) -> dict:
    url = f"{SPACE_URLS[task]}/web/step"
    r = requests.post(url, json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

CLEANING_SYSTEM = """You are a data quality agent. You are given a data observation showing quality issues.
Output ONLY a JSON action to fix one issue. No explanation, just the JSON.
Format: {"operation": "<op>", "column": "<col>", "value": "<val_or_null>", "params": {}}
Operations: fill_null, fix_type, remove_duplicate, standardize, trim, correct_typo
For fill_null: use value "median" for numeric columns, "mode" for categorical.
For standardize: use value "lower" or "title".
For remove_duplicate: column can be any column.
ONLY output the JSON object, nothing else."""

ENRICHMENT_SYSTEM = """You are a data enrichment agent. You are given a dataset schema and possible enrichments.
Output ONLY a JSON action to add a derived field. No explanation, just the JSON.
Format: {"operation": "<op>", "field_name": "<name>", "params": {}}
Operations: add_field, lookup, compute_derived, add_category
Pick from the possible_enrichments list.
ONLY output the JSON object, nothing else."""

ANSWERING_SYSTEM = """You are a data analyst answering questions for a specific persona.
Adapt your language and focus to match the persona's style:
- Executive: strategic, financial, ROI-focused, use metrics and percentages
- Manager: operational, actionable, focus on team performance and processes
- Individual Contributor: plain language, personal, focus on next steps and tasks
Cite specific data columns in your answer. Be data-grounded and avoid speculation."""


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_json_action(response: str) -> dict:
    """Extract JSON from model response."""
    # Try to find JSON block
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_cleaning_episode(model_key: str) -> dict:
    """Run one cleaning episode. Returns metrics dict."""
    reset_data = env_reset("cleaning")
    obs = reset_data["observation"]
    initial_dq = obs["dq_score"]
    domain = obs["domain"]

    total_reward = 0.0
    steps = 0
    max_steps = obs.get("max_steps", 15)
    done = False
    actions_log = []

    while not done and steps < max_steps:
        # Build prompt from observation
        prompt = (
            f"Domain: {obs['domain']}\n"
            f"DQ Score: {obs['dq_score']}\n"
            f"DQ Report: {obs['dq_report']}\n"
            f"Columns:\n{obs['columns_info']}\n"
            f"Data Preview:\n{obs['data_preview'][:500]}\n"
            f"Step {obs['step_number']}/{obs['max_steps']}\n\n"
            "Generate a JSON action to fix the most impactful issue."
        )

        try:
            response = call_model(model_key, prompt, CLEANING_SYSTEM)
            action = parse_json_action(response)
            if not action or "operation" not in action:
                action = {"operation": "trim", "column": "Status", "params": {}}
        except Exception as e:
            action = {"operation": "trim", "column": "Status", "params": {}}

        # Ensure required fields
        action.setdefault("params", {})
        actions_log.append(action)

        try:
            result = env_step("cleaning", action)
            obs = result["observation"]
            total_reward += result.get("reward", 0)
            done = result.get("done", False)
            steps += 1
        except Exception as e:
            print(f"    Step error: {e}")
            break

    final_dq = obs.get("dq_score", initial_dq)
    return {
        "domain": domain,
        "initial_dq": round(initial_dq, 4),
        "final_dq": round(final_dq, 4),
        "dq_improvement": round(final_dq - initial_dq, 4),
        "total_reward": round(total_reward, 4),
        "avg_reward": round(total_reward / max(steps, 1), 4),
        "steps": steps,
        "actions": actions_log,
    }


def run_enrichment_episode(model_key: str) -> dict:
    """Run one enrichment episode."""
    reset_data = env_reset("enrichment")
    obs = reset_data["observation"]
    domain = obs["domain"]
    initial_coverage = obs.get("enrichment_coverage", 0)

    total_reward = 0.0
    steps = 0
    max_steps = obs.get("max_steps", 12)
    done = False
    actions_log = []

    while not done and steps < max_steps:
        prompt = (
            f"Domain: {obs['domain']}\n"
            f"Schema:\n{obs['schema_info']}\n"
            f"Current Coverage: {obs.get('enrichment_coverage', 0)}\n"
            f"Fields Added: {obs.get('fields_added', [])}\n"
            f"Possible Enrichments: {obs.get('possible_enrichments', [])}\n"
            f"Step {obs['step_number']}/{obs['max_steps']}\n\n"
            "Generate a JSON action to add the most valuable enrichment."
        )

        try:
            response = call_model(model_key, prompt, ENRICHMENT_SYSTEM)
            action = parse_json_action(response)
            if not action or "operation" not in action:
                possible = obs.get("possible_enrichments", ["unknown"])
                action = {"operation": "add_field", "field_name": possible[0], "params": {}}
        except Exception as e:
            possible = obs.get("possible_enrichments", ["unknown"])
            action = {"operation": "add_field", "field_name": possible[0], "params": {}}

        action.setdefault("params", {})
        actions_log.append(action)

        try:
            result = env_step("enrichment", action)
            obs = result["observation"]
            total_reward += result.get("reward", 0)
            done = result.get("done", False)
            steps += 1
        except Exception as e:
            print(f"    Step error: {e}")
            break

    final_coverage = obs.get("enrichment_coverage", 0)
    return {
        "domain": domain,
        "initial_coverage": round(initial_coverage, 4),
        "final_coverage": round(final_coverage, 4),
        "total_reward": round(total_reward, 4),
        "avg_reward": round(total_reward / max(steps, 1), 4),
        "steps": steps,
        "fields_added": len(actions_log),
        "actions": actions_log,
    }


def run_answering_episode(model_key: str) -> dict:
    """Run one answering episode."""
    reset_data = env_reset("answering")
    obs = reset_data["observation"]
    domain = obs["domain"]
    persona = obs["persona"]
    question = obs["question"]

    prompt = (
        f"Domain: {obs['domain']}\n"
        f"Persona: {obs['persona']} - {obs.get('persona_description', '')}\n"
        f"Question: {obs['question']}\n"
        f"Available Columns: {obs.get('available_columns', [])}\n"
        f"Column Stats:\n{obs.get('column_stats', '')[:800]}\n"
        f"Dataset Summary:\n{obs.get('dataset_summary', '')[:400]}\n\n"
        "Answer the question in the style appropriate for this persona."
    )

    try:
        response = call_model(model_key, prompt, ANSWERING_SYSTEM)
    except Exception as e:
        response = f"Unable to generate answer: {e}"

    # Extract cited columns from the answer
    available = obs.get("available_columns", [])
    cited = [c for c in available if c.lower() in response.lower()]

    action = {
        "answer": response,
        "cited_columns": cited,
        "reasoning": "",
    }

    try:
        result = env_step("answering", action)
        reward = result.get("reward", 0)
        done = result.get("done", True)
    except Exception as e:
        print(f"    Step error: {e}")
        reward = 0
        done = True

    return {
        "domain": domain,
        "persona": persona,
        "question": question,
        "answer": response[:500],
        "cited_columns": cited,
        "reward": round(reward, 4),
        "done": done,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmarks():
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_episodes": N_EPISODES,
            "models": list(MODELS_CONFIG.keys()),
        },
        "cleaning": {},
        "enrichment": {},
        "answering": {},
    }

    for model_key in MODELS_CONFIG:
        display = MODELS_CONFIG[model_key]["display_name"]
        print(f"\n{'='*60}")
        print(f"  Model: {display}")
        print(f"{'='*60}")

        # --- CLEANING ---
        print(f"\n  [Cleaning]")
        cleaning_episodes = []
        for ep in range(N_EPISODES):
            print(f"    Episode {ep+1}/{N_EPISODES}...", end=" ", flush=True)
            try:
                metrics = run_cleaning_episode(model_key)
                # Remove actions to keep output clean
                actions = metrics.pop("actions", [])
                cleaning_episodes.append(metrics)
                print(f"domain={metrics['domain']} dq={metrics['initial_dq']:.4f}->{metrics['final_dq']:.4f} "
                      f"(+{metrics['dq_improvement']:.4f}) steps={metrics['steps']}")
            except Exception as e:
                print(f"FAILED: {e}")
                cleaning_episodes.append({"error": str(e)})

        results["cleaning"][model_key] = {
            "episodes": cleaning_episodes,
            "summary": _summarize(cleaning_episodes, ["final_dq", "dq_improvement", "avg_reward", "steps"]),
        }

        # --- ENRICHMENT ---
        print(f"\n  [Enrichment]")
        enrichment_episodes = []
        for ep in range(N_EPISODES):
            print(f"    Episode {ep+1}/{N_EPISODES}...", end=" ", flush=True)
            try:
                metrics = run_enrichment_episode(model_key)
                actions = metrics.pop("actions", [])
                enrichment_episodes.append(metrics)
                print(f"domain={metrics['domain']} coverage={metrics['final_coverage']:.4f} "
                      f"steps={metrics['steps']}")
            except Exception as e:
                print(f"FAILED: {e}")
                enrichment_episodes.append({"error": str(e)})

        results["enrichment"][model_key] = {
            "episodes": enrichment_episodes,
            "summary": _summarize(enrichment_episodes, ["final_coverage", "avg_reward", "steps", "fields_added"]),
        }

        # --- ANSWERING ---
        print(f"\n  [Answering]")
        answering_episodes = []
        for ep in range(N_EPISODES):
            print(f"    Episode {ep+1}/{N_EPISODES}...", end=" ", flush=True)
            try:
                metrics = run_answering_episode(model_key)
                answering_episodes.append(metrics)
                print(f"domain={metrics['domain']} persona={metrics['persona']} "
                      f"reward={metrics['reward']:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")
                answering_episodes.append({"error": str(e)})

        results["answering"][model_key] = {
            "episodes": answering_episodes,
            "summary": _summarize(answering_episodes, ["reward"]),
        }

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "data", "real_benchmark_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for model_key in MODELS_CONFIG:
        display = MODELS_CONFIG[model_key]["display_name"]
        c = results["cleaning"].get(model_key, {}).get("summary", {})
        e = results["enrichment"].get(model_key, {}).get("summary", {})
        a = results["answering"].get(model_key, {}).get("summary", {})
        print(f"\n  {display}:")
        print(f"    Cleaning:   DQ={c.get('final_dq_mean', 0):.4f} improvement={c.get('dq_improvement_mean', 0):.4f}")
        print(f"    Enrichment: coverage={e.get('final_coverage_mean', 0):.4f}")
        print(f"    Answering:  reward={a.get('reward_mean', 0):.4f}")

    return results


def _summarize(episodes: list, keys: list) -> dict:
    """Compute mean/std for numeric keys across episodes."""
    summary = {}
    valid = [ep for ep in episodes if "error" not in ep]
    if not valid:
        return summary
    for key in keys:
        values = [ep.get(key, 0) for ep in valid if isinstance(ep.get(key), (int, float))]
        if values:
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5
            summary[f"{key}_mean"] = round(mean, 4)
            summary[f"{key}_std"] = round(std, 4)
    summary["n_valid"] = len(valid)
    summary["n_total"] = len(episodes)
    return summary


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    run_benchmarks()
