"""Smoke tests for the cleaning environment."""

import sys
import os

# Add project root first so `environments.*` resolves correctly.
# The cleaning dir is *appended* (not inserted at 0) to avoid its local
# ``environments/`` sub-directory shadowing the top-level package.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_cleaning_root = os.path.join(_project_root, 'environments', 'cleaning')
if _cleaning_root not in sys.path:
    sys.path.append(_cleaning_root)

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


def test_cleaning_seeded_reset():
    """Seeded resets must produce identical state."""
    env1 = CleaningEnvironment()
    obs1 = env1.reset(seed=42, domain="hr")

    env2 = CleaningEnvironment()
    obs2 = env2.reset(seed=42, domain="hr")

    assert obs1.domain == obs2.domain == "hr"
    assert obs1.dq_score == obs2.dq_score, f"DQ mismatch: {obs1.dq_score} vs {obs2.dq_score}"
    assert obs1.data_preview == obs2.data_preview, "Data preview should be identical"
    print(f"PASS: seeded reset produces identical state, dq={obs1.dq_score:.4f}")


def test_cleaning_seeded_reset_different_seeds():
    """Different seeds must produce different state."""
    env1 = CleaningEnvironment()
    obs1 = env1.reset(seed=42, domain="hr")

    env2 = CleaningEnvironment()
    obs2 = env2.reset(seed=99, domain="hr")

    assert obs1.dq_score != obs2.dq_score or obs1.data_preview != obs2.data_preview, \
        "Different seeds should produce different states"
    print(f"PASS: different seeds produce different states")


if __name__ == "__main__":
    test_cleaning_reset()
    test_cleaning_step()
    test_cleaning_all_domains()
    test_cleaning_seeded_reset()
    test_cleaning_seeded_reset_different_seeds()
    print("\nAll cleaning tests PASSED")
