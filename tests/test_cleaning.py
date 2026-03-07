"""Smoke tests for the cleaning environment."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environments', 'cleaning'))

from environments.cleaning.server.cleaning_environment import CleaningEnvironment
from environments.cleaning.models import CleaningAction


def test_cleaning_reset():
    env = CleaningEnvironment()
    obs = env.reset()
    assert obs.domain in ["hr", "sales", "pm", "it_ops"], f"Unexpected domain: {obs.domain}"
    assert obs.dq_score > 0, "DQ score should be positive"
    assert obs.dq_score < 1.0 or obs.dq_score == 1.0, "DQ should be valid"
    assert obs.step_number == 0
    assert obs.max_steps == 15
    assert obs.done is False
    assert len(obs.data_preview) > 0
    assert len(obs.columns_info) > 0
    print(f"PASS: reset() domain={obs.domain} dq={obs.dq_score:.4f}")
    return obs


def test_cleaning_step():
    env = CleaningEnvironment()
    obs = env.reset()

    # Try fill_null on a numeric column
    config_cols = obs.columns_info.split("\n")
    target_col = None
    for line in config_cols:
        if "nulls=" in line and "nulls=0" not in line:
            target_col = line.split(":")[0].strip()
            break

    if target_col is None:
        # Fallback: just use first column from domain
        from environments.shared.domains import DOMAINS
        target_col = DOMAINS[obs.domain].numeric_columns[0]

    action = CleaningAction(operation="fill_null", column=target_col, value="median")
    obs2 = env.step(action)
    assert obs2.reward >= 0, f"Reward should be non-negative: {obs2.reward}"
    assert obs2.step_number == 1
    print(f"PASS: step(fill_null) domain={obs2.domain} dq={obs2.dq_score:.4f} reward={obs2.reward:.4f}")

    # Try remove_duplicate
    action2 = CleaningAction(operation="remove_duplicate", column="")
    obs3 = env.step(action2)
    assert obs3.step_number == 2
    print(f"PASS: step(remove_duplicate) dq={obs3.dq_score:.4f} reward={obs3.reward:.4f}")

    return obs3


def test_cleaning_all_domains():
    env = CleaningEnvironment()
    domains_seen = set()
    for _ in range(20):
        obs = env.reset()
        domains_seen.add(obs.domain)
        if len(domains_seen) == 4:
            break
    print(f"PASS: domains seen: {domains_seen}")
    assert len(domains_seen) >= 2, f"Should see multiple domains, got: {domains_seen}"


if __name__ == "__main__":
    test_cleaning_reset()
    test_cleaning_step()
    test_cleaning_all_domains()
    print("\nAll cleaning tests PASSED")
