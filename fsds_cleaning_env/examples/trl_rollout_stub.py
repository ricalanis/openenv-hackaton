"""Minimal TRL rollout stub for OpenEnv integration.

This follows the official TRL + OpenEnv pattern at a high level: generate a completion,
parse it into a tool call, step the environment, then forward env_reward into reward_fn.
Replace the placeholder generation/parsing logic with your actual model sampling code.
"""

from __future__ import annotations

import json
from typing import Any

from fsds_cleaning_env import FSDSCleaningEnv


def parse_completion_to_tool_call(text: str) -> dict[str, Any]:
    """Expect a JSON object like:
    {"tool": "apply_cleaning_operation", "arguments": {"operation": "drop_duplicates"}}
    """
    try:
        data = json.loads(text)
        return data
    except Exception:
        return {
            "tool": "run_quality_gates",
            "arguments": {},
        }


def rollout_func(prompts: list[str], trainer) -> dict[str, list]:
    prompt_ids = []
    completion_ids = []
    logprobs = []
    env_rewards = []

    # Replace with colocated vLLM generation or trainer.generate equivalents.
    fake_completions = [
        '{"tool": "apply_cleaning_operation", "arguments": {"operation": "drop_duplicates"}}'
        for _ in prompts
    ]

    with FSDSCleaningEnv(base_url="https://YOUR-SPACE.hf.space").sync() as env:
        for prompt, completion in zip(prompts, fake_completions):
            del prompt
            env.reset(task_id="ecommerce_mobile")
            tool_call = parse_completion_to_tool_call(completion)
            result = env.call_tool(tool_call["tool"], **tool_call.get("arguments", {}))
            reward = float(result.get("reward", 0.0))
            env_rewards.append(reward)

            # Replace these placeholders with tokenizer-accurate values from your generation path.
            prompt_ids.append([0])
            completion_ids.append([0])
            logprobs.append([0.0])

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_reward": env_rewards,
    }


def reward_fn(prompts, completions, env_reward=None, **kwargs):
    del prompts, completions, kwargs
    return env_reward
