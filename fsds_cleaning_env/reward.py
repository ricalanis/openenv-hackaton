from __future__ import annotations

from dataclasses import dataclass


# Default rewards used when a tool call fails at the environment level.
TOOL_ERROR_REWARD: float = -0.2

# Default shaping parameters for per-step rewards.
STEP_IMPROVEMENT_MARGIN: float = 0.02
STEP_MAX_NEGATIVE: float = -0.15


@dataclass
class StepRewardInput:
    """Inputs required to compute a per-step reward.

    The FSDS Cleaning Environment currently uses a dense shaping term based on
    quality-score improvement, with a small step cost and a floor on negative
    rewards to keep trajectories numerically stable.
    """

    quality_before: float
    quality_after: float
    improvement_margin: float = STEP_IMPROVEMENT_MARGIN
    max_negative: float = STEP_MAX_NEGATIVE


def compute_step_reward(inp: StepRewardInput) -> float:
    """Reward for a single cleaning operation.

    Formula (before clipping):
        (quality_after - quality_before) - improvement_margin

    Then clipped from below by max_negative to avoid excessively large
    negative values on obviously bad actions.
    """

    delta = float(inp.quality_after) - float(inp.quality_before)
    raw = delta - float(inp.improvement_margin)
    return max(float(inp.max_negative), float(raw))


def compute_quality_gate_bonus(passed: bool) -> float:
    """Reward bonus/penalty for running quality gates."""

    return 0.15 if passed else -0.1


@dataclass
class FinalRewardInput:
    """Inputs for the terminal reward at submission time."""

    quality_score: float
    gate_passed: bool
    required_operation_coverage: float


def compute_final_reward(inp: FinalRewardInput) -> float:
    """Combine table quality, gate result, and required-op coverage."""

    gate_term = 1.0 if inp.gate_passed else 0.0
    score = (
        0.45 * float(inp.quality_score)
        + 0.30 * gate_term
        + 0.25 * float(inp.required_operation_coverage)
    )
    return float(score)


__all__ = [
    "TOOL_ERROR_REWARD",
    "STEP_IMPROVEMENT_MARGIN",
    "STEP_MAX_NEGATIVE",
    "StepRewardInput",
    "compute_step_reward",
    "compute_quality_gate_bonus",
    "FinalRewardInput",
    "compute_final_reward",
]

