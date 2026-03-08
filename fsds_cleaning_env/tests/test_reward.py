from __future__ import annotations

from fsds_cleaning_env.reward import (
    FinalRewardInput,
    StepRewardInput,
    TOOL_ERROR_REWARD,
    compute_final_reward,
    compute_quality_gate_bonus,
    compute_step_reward,
)


def test_step_reward_positive_improvement() -> None:
    inp = StepRewardInput(quality_before=0.5, quality_after=0.7)
    reward = compute_step_reward(inp)
    # Delta = 0.2, minus margin 0.02 => 0.18
    assert abs(reward - 0.18) < 1e-6


def test_step_reward_negative_clipped() -> None:
    inp = StepRewardInput(quality_before=0.5, quality_after=0.2)
    reward = compute_step_reward(inp)
    # Delta = -0.3, minus margin 0.02 => -0.32, clipped to -0.15
    assert abs(reward - (-0.15)) < 1e-6


def test_quality_gate_bonus() -> None:
    assert compute_quality_gate_bonus(True) == 0.15
    assert compute_quality_gate_bonus(False) == -0.1


def test_final_reward_combination() -> None:
    inp = FinalRewardInput(
        quality_score=1.0,
        gate_passed=True,
        required_operation_coverage=1.0,
    )
    reward = compute_final_reward(inp)
    # 0.45 * 1.0 + 0.30 * 1.0 + 0.25 * 1.0 = 1.0
    assert abs(reward - 1.0) < 1e-6


def test_tool_error_reward_constant() -> None:
    assert TOOL_ERROR_REWARD < 0.0

