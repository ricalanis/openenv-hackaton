"""Demonstration collection and SFT dataset building for the FSDS Cleaning Environment.

Phase 7 — SFT-first, RL-second
================================
Expert trajectories are collected via the HeuristicAgent (or any scripted policy)
and serialized to JSON.  They are then converted into two SFT formats:

  step   — one training example per action step (N examples per episode)
  episode — one multi-turn conversation per episode (1 example per episode)

Both formats use the same system prompt as the GRPO training in training_colab.py
so that the SFT checkpoint is a natural warm-start for RL fine-tuning.

Typical usage
-------------
Offline (no server, for testing):
    from fsds_cleaning_env.demonstrations import build_sft_dataset_from_heuristic
    dataset = build_sft_dataset_from_heuristic()

With a live server:
    from fsds_cleaning_env.demonstrations import (
        DemonstrationCollector, build_sft_dataset, save_demonstrations
    )
    with FSDSCleaningEnv(base_url=...).sync() as env:
        demos = DemonstrationCollector(env).collect(
            task_ids=["ecommerce_mobile", "subscription_churn", "delivery_eta"],
            n_per_task=20,
        )
    save_demonstrations(demos, "demos/expert_demos.json")
    dataset = build_sft_dataset(demos, mode="step")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional


# ── System prompt (mirrors training_colab.py) ──────────────────────────────────

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
  {"tool": "run_quality_gates", "arguments": {}}
  {"tool": "submit_solution", "arguments": {}}

Think step by step. Inspect before cleaning. Validate before submitting."""


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class DemoStep:
    """One action step within a demonstration episode."""

    step_idx: int
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    reward: float


@dataclass
class Demonstration:
    """A full expert episode trajectory."""

    task_id: str
    seed: Optional[int]
    steps: List[DemoStep]
    total_reward: float
    success: bool
    difficulty: str = "medium"  # easy | medium | hard — from CurriculumTask if used

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "total_reward": self.total_reward,
            "success": self.success,
            "steps": [asdict(s) for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Demonstration":
        steps = [DemoStep(**s) for s in d.get("steps", [])]
        return cls(
            task_id=d["task_id"],
            seed=d.get("seed"),
            steps=steps,
            total_reward=d.get("total_reward", 0.0),
            success=d.get("success", False),
            difficulty=d.get("difficulty", "medium"),
        )


# ── Serialization ──────────────────────────────────────────────────────────────

def save_demonstrations(demos: List[Demonstration], path: str | Path) -> None:
    """Persist demonstrations to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump([d.to_dict() for d in demos], f, indent=2)
    print(f"Saved {len(demos)} demonstrations to {p}")


def load_demonstrations(path: str | Path) -> List[Demonstration]:
    """Load demonstrations from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    demos = [Demonstration.from_dict(d) for d in data]
    print(f"Loaded {len(demos)} demonstrations from {path}")
    return demos


# ── Collection ─────────────────────────────────────────────────────────────────

class DemonstrationCollector:
    """Collect expert trajectories from any agent that follows the Agent protocol."""

    def __init__(self, env: Any) -> None:
        self._env = env

    def collect(
        self,
        task_ids: Optional[List[str]] = None,
        n_per_task: int = 10,
        seed_offset: int = 0,
        max_steps: int = 18,
        difficulty: str = "medium",
        noise_profile: Any = None,
        n_rows: Optional[int] = None,
    ) -> List[Demonstration]:
        """Run the HeuristicAgent and return collected demonstrations.

        Parameters
        ----------
        task_ids:
            Which tasks to collect from. Defaults to all three built-in tasks.
        n_per_task:
            Episodes per task.
        seed_offset:
            Starting seed; each episode uses ``seed_offset + episode_index``.
        max_steps:
            Step budget forwarded to agent.run_episode().
        difficulty:
            Label recorded in each Demonstration (informational).
        noise_profile:
            Optional NoiseProfile to pass as ``noise_profile_override`` to reset.
        n_rows:
            Optional row count override for each episode.
        """
        from fsds_cleaning_env.agents import HeuristicAgent

        if task_ids is None:
            task_ids = ["ecommerce_mobile", "subscription_churn", "delivery_eta"]

        agent = HeuristicAgent()
        demos: List[Demonstration] = []

        for task_id in task_ids:
            for i in range(n_per_task):
                seed = seed_offset + i
                reset_kwargs: dict[str, Any] = {}
                if noise_profile is not None:
                    reset_kwargs["noise_profile_override"] = noise_profile
                if n_rows is not None:
                    reset_kwargs["dataset_n_rows"] = n_rows

                trajectory = agent.run_episode(
                    self._env,
                    task_id=task_id,
                    max_steps=max_steps,
                    seed=seed,
                    **reset_kwargs,
                )

                steps = []
                total_reward = 0.0
                success = False
                for idx, step in enumerate(trajectory):
                    result = step.get("result", {})
                    # Extract arguments from result metadata (best-effort).
                    args = _infer_arguments(step.get("tool_name", ""), result)
                    reward = float(step.get("reward", 0.0))
                    total_reward += reward
                    if result.get("done", False):
                        success = result.get("final_reward", 0.0) > 0.5
                    steps.append(
                        DemoStep(
                            step_idx=idx,
                            tool_name=step.get("tool_name", ""),
                            arguments=args,
                            result=result,
                            reward=reward,
                        )
                    )

                demos.append(
                    Demonstration(
                        task_id=task_id,
                        seed=seed,
                        steps=steps,
                        total_reward=total_reward,
                        success=success,
                        difficulty=difficulty,
                    )
                )

        print(
            f"Collected {len(demos)} demonstrations across {len(task_ids)} tasks "
            f"({n_per_task} per task). "
            f"Success rate: {sum(d.success for d in demos) / len(demos):.1%}"
        )
        return demos


def _infer_arguments(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Best-effort: reconstruct the action arguments from the trajectory step.

    The HeuristicAgent builds kwargs before calling env.call_tool, but the raw
    trajectory only stores the result.  We recover what we can from ``result``
    metadata (operation_log entries, applied_operation field, etc.).
    """
    if tool_name != "apply_cleaning_operation":
        return {}
    # Some env versions echo the applied operation in the result dict.
    if "applied_operation" in result:
        op = result["applied_operation"]
        col = result.get("column")
        args: dict[str, Any] = {"operation": op}
        if col:
            args["column"] = col
        return args
    return {}


# ── SFT formatting ─────────────────────────────────────────────────────────────

def _action_to_json(tool_name: str, arguments: dict[str, Any]) -> str:
    """Render a tool call as the JSON string the model should emit."""
    return json.dumps({"tool": tool_name, "arguments": arguments}, ensure_ascii=False)


def _result_summary(result: dict[str, Any]) -> str:
    """Render an environment result as a concise observation string."""
    # Drop large fields that would inflate token count.
    filtered = {
        k: v
        for k, v in result.items()
        if k not in ("preview", "sample_rows", "data") and not isinstance(v, list)
    }
    return json.dumps(filtered, ensure_ascii=False)


def demo_to_step_examples(
    demo: Demonstration,
    system_prompt: str = SYSTEM_PROMPT,
) -> List[dict[str, Any]]:
    """Convert one demonstration into N step-level SFT examples.

    Each example is a dict with keys ``"prompt"`` (list of message dicts) and
    ``"completion"`` (the JSON action string the model should predict).

    Step k's prompt contains:
      system | user: task context + formatted history of steps 0..k-1

    This format matches the GRPO dataset in training_colab.py so the same
    tokenisation pipeline works for both SFT and GRPO.
    """
    examples = []
    history_lines: List[str] = [f"Task ID: {demo.task_id}"]

    for step in demo.steps:
        # Build user message from accumulated context.
        user_content = "\n".join(history_lines) if history_lines else f"Task ID: {demo.task_id}"
        if step.step_idx == 0:
            user_content = (
                f"Task: {demo.task_id}\n"
                "You have just received a dirty dataset. "
                "Inspect it and begin the cleaning pipeline."
            )

        completion = _action_to_json(step.tool_name, step.arguments)

        examples.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "completion": completion,
                "task_id": demo.task_id,
                "step_idx": step.step_idx,
                "reward": step.reward,
                "difficulty": demo.difficulty,
            }
        )

        # Append this step's action + result to history for the next step.
        obs_summary = _result_summary(step.result)
        history_lines.append(f"Action {step.step_idx}: {completion}")
        history_lines.append(f"Observation {step.step_idx}: {obs_summary}")

    return examples


def demo_to_episode_example(
    demo: Demonstration,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Convert one demonstration into a single multi-turn conversation.

    The conversation alternates:
        assistant: <JSON action>
        user:      <environment result summary>

    Starting user message describes the task.
    """
    messages: List[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Task: {demo.task_id}\n"
                "You have just received a dirty dataset. "
                "Inspect it and begin the cleaning pipeline."
            ),
        },
    ]

    for step in demo.steps:
        action_json = _action_to_json(step.tool_name, step.arguments)
        messages.append({"role": "assistant", "content": action_json})
        obs = _result_summary(step.result)
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return {
        "messages": messages,
        "task_id": demo.task_id,
        "total_reward": demo.total_reward,
        "success": demo.success,
        "difficulty": demo.difficulty,
    }


# ── Dataset builders ───────────────────────────────────────────────────────────

def build_sft_dataset(
    demos: List[Demonstration],
    mode: str = "step",
    system_prompt: str = SYSTEM_PROMPT,
    successful_only: bool = True,
) -> "datasets.Dataset":  # type: ignore[name-defined]
    """Build a Hugging Face Dataset from collected demonstrations.

    Parameters
    ----------
    demos:
        List of Demonstration objects (from DemonstrationCollector or loaded from disk).
    mode:
        ``"step"``    — one row per action step (N rows × episodes).
        ``"episode"`` — one row per episode as a multi-turn conversation.
    system_prompt:
        The system prompt injected into every example.
    successful_only:
        If True (default), filter to demonstrations where success=True so the
        model only learns from winning trajectories.
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets library required: pip install datasets")

    if successful_only:
        before = len(demos)
        demos = [d for d in demos if d.success]
        print(f"Filtered to successful demonstrations: {len(demos)}/{before}")

    if mode == "step":
        rows = []
        for demo in demos:
            rows.extend(demo_to_step_examples(demo, system_prompt=system_prompt))
        print(f"Built step-level SFT dataset: {len(rows)} examples from {len(demos)} episodes")
        return Dataset.from_list(rows)

    if mode == "episode":
        rows = [demo_to_episode_example(demo, system_prompt=system_prompt) for demo in demos]
        print(f"Built episode-level SFT dataset: {len(rows)} examples")
        return Dataset.from_list(rows)

    raise ValueError(f"Unknown mode: {mode!r}. Use 'step' or 'episode'.")


def build_sft_dataset_from_heuristic(
    task_ids: Optional[List[str]] = None,
    n_per_task: int = 5,
    mode: str = "step",
    system_prompt: str = SYSTEM_PROMPT,
    base_url: str = "http://localhost:8000",
) -> "datasets.Dataset":  # type: ignore[name-defined]
    """Convenience function: spin up a client, collect demonstrations, build dataset.

    Requires a running environment server at ``base_url``.
    """
    from fsds_cleaning_env.client import FSDSCleaningEnv

    with FSDSCleaningEnv(base_url=base_url).sync() as env:
        collector = DemonstrationCollector(env)
        demos = collector.collect(
            task_ids=task_ids,
            n_per_task=n_per_task,
        )

    return build_sft_dataset(demos, mode=mode, system_prompt=system_prompt)


# ── Statistics ─────────────────────────────────────────────────────────────────

def demo_stats(demos: List[Demonstration]) -> dict[str, Any]:
    """Return summary statistics for a list of demonstrations."""
    if not demos:
        return {}
    total = len(demos)
    successful = sum(d.success for d in demos)
    avg_steps = sum(len(d.steps) for d in demos) / total
    avg_reward = sum(d.total_reward for d in demos) / total
    by_task: dict[str, dict[str, Any]] = {}
    for d in demos:
        t = d.task_id
        if t not in by_task:
            by_task[t] = {"count": 0, "success": 0, "total_reward": 0.0}
        by_task[t]["count"] += 1
        by_task[t]["success"] += int(d.success)
        by_task[t]["total_reward"] += d.total_reward
    return {
        "total": total,
        "success_rate": successful / total,
        "avg_steps": round(avg_steps, 2),
        "avg_reward": round(avg_reward, 4),
        "by_task": {
            t: {
                "count": v["count"],
                "success_rate": v["success"] / v["count"],
                "avg_reward": round(v["total_reward"] / v["count"], 4),
            }
            for t, v in by_task.items()
        },
    }


__all__ = [
    "SYSTEM_PROMPT",
    "DemoStep",
    "Demonstration",
    "DemonstrationCollector",
    "save_demonstrations",
    "load_demonstrations",
    "demo_to_step_examples",
    "demo_to_episode_example",
    "build_sft_dataset",
    "build_sft_dataset_from_heuristic",
    "demo_stats",
]
