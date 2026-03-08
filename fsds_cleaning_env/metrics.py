from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


Step = Dict[str, Any]


@dataclass
class EpisodeMetrics:
    """Summary statistics for a single episode trajectory.

    This is intentionally environment-agnostic and works off a generic
    trajectory structure:

    - Each step is a dict with at least:
        - "reward": float
        - "done": bool
        - "result": optional dict (tool result / observation payload)
        - "tool_name": optional str
    """

    success: bool
    total_return: float
    steps: int
    invalid_actions: int
    quality_gate_passed: Optional[bool]
    retention_ratio: Optional[float]


@dataclass
class AggregateMetrics:
    """Aggregated metrics over many episodes."""

    episodes: int
    success_rate: float
    avg_return: float
    avg_steps: float
    avg_invalid_actions: float


def compute_episode_metrics(trajectory: Iterable[Step]) -> EpisodeMetrics:
    """Compute metrics for a single episode.

    Parameters
    ----------
    trajectory:
        Iterable of step dicts. Each step should contain:
        - "reward": float-compatible value
        - "done": bool (unused for now but kept for completeness)
        - "result": optional dict returned by the environment/tool call

    The FSDS Cleaning Environment typically returns:
    - For cleaning steps: dict with "reward", "quality_score", "shape"
    - For `run_quality_gates`: dict with "passed", "tests", "reward", "total_reward", "retention_ratio"
    - For `submit_solution`: dict with "done", "passed", "final_reward",
      "cumulative_reward", "quality_report", "required_operation_coverage", ...
    - For invalid actions: dict with "error" and a negative "reward".
    """

    steps_list: List[Step] = list(trajectory)
    total_return = float(sum(float(step.get("reward", 0.0)) for step in steps_list))
    steps = len(steps_list)

    invalid_actions = 0
    last_result: Optional[Dict[str, Any]] = None

    for step in steps_list:
        result = step.get("result")
        if isinstance(result, dict):
            last_result = result
            if "error" in result:
                invalid_actions += 1

    quality_gate_passed: Optional[bool] = None
    retention_ratio: Optional[float] = None
    success = False

    if isinstance(last_result, dict):
        # Direct fields from `run_quality_gates` or `submit_solution`
        if "passed" in last_result:
            quality_gate_passed = bool(last_result.get("passed"))
            success = quality_gate_passed
        if "retention_ratio" in last_result:
            try:
                retention_ratio = float(last_result.get("retention_ratio"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                retention_ratio = None

        # Nested quality report from `submit_solution`
        quality_report = last_result.get("quality_report")
        if isinstance(quality_report, dict):
            if "passed" in quality_report:
                quality_gate_passed = bool(quality_report.get("passed"))
                success = quality_gate_passed
            if "retention_ratio" in quality_report:
                try:
                    retention_ratio = float(quality_report.get("retention_ratio"))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    retention_ratio = retention_ratio

    return EpisodeMetrics(
        success=success,
        total_return=total_return,
        steps=steps,
        invalid_actions=invalid_actions,
        quality_gate_passed=quality_gate_passed,
        retention_ratio=retention_ratio,
    )


def aggregate_metrics(episodes: Iterable[EpisodeMetrics]) -> AggregateMetrics:
    """Aggregate metrics over many episodes."""

    ep_list = list(episodes)
    n = len(ep_list)
    if n == 0:
        return AggregateMetrics(
            episodes=0,
            success_rate=0.0,
            avg_return=0.0,
            avg_steps=0.0,
            avg_invalid_actions=0.0,
        )

    success_rate = sum(1 for ep in ep_list if ep.success) / n
    avg_return = sum(ep.total_return for ep in ep_list) / n
    avg_steps = sum(ep.steps for ep in ep_list) / n
    avg_invalid_actions = sum(ep.invalid_actions for ep in ep_list) / n

    return AggregateMetrics(
        episodes=n,
        success_rate=float(success_rate),
        avg_return=float(avg_return),
        avg_steps=float(avg_steps),
        avg_invalid_actions=float(avg_invalid_actions),
    )


__all__ = ["Step", "EpisodeMetrics", "AggregateMetrics", "compute_episode_metrics", "aggregate_metrics"]

