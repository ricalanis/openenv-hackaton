"""Smoke tests for the answering environment."""

import sys
import os

# Add project root first so `environments.*` resolves correctly.
# The answering dir is *appended* (not inserted at 0) to avoid its local
# ``environments/`` sub-directory shadowing the top-level package.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_answering_root = os.path.join(_project_root, 'environments', 'answering')
if _answering_root not in sys.path:
    sys.path.append(_answering_root)

from environments.answering.server.answering_environment import AnsweringEnvironment
from environments.answering.client import AnsweringEnv
from environments.answering.models import AnsweringAction


def test_answering_reset():
    env = AnsweringEnvironment()
    obs = env.reset()
    assert obs.domain in ["hr", "sales", "pm", "it_ops"]
    assert obs.persona in ["Executive", "Manager", "Individual Contributor"]
    assert len(obs.question) > 0
    assert len(obs.available_columns) > 0
    assert obs.done is False
    print(f"PASS: reset() domain={obs.domain} persona={obs.persona} question={obs.question[:60]}...")
    return obs


def test_answering_step_good():
    env = AnsweringEnvironment()
    obs = env.reset()

    action = AnsweringAction(
        answer=f"Based on analysis of {obs.available_columns[0]} and {obs.available_columns[1]}, "
               f"the team performance shows strong trends with cost reduction impact. "
               f"The recommended action is to prioritize high-risk items for immediate review.",
        cited_columns=obs.available_columns[:3],
        reasoning="Analyzed the data columns to identify key patterns and trends."
    )
    obs2 = env.step(action)
    assert obs2.done is True
    assert obs2.reward > 0
    print(f"PASS: good answer reward={obs2.reward:.4f} done={obs2.done}")
    return obs2


def test_answering_step_bad():
    env = AnsweringEnvironment()
    obs = env.reset()

    action = AnsweringAction(
        answer="I don't know.",
        cited_columns=[],
        reasoning=""
    )
    obs2 = env.step(action)
    assert obs2.done is True
    print(f"PASS: bad answer reward={obs2.reward:.4f} (should be low)")
    return obs2


def test_answering_all_personas():
    env = AnsweringEnvironment()
    personas_seen = set()
    for _ in range(20):
        obs = env.reset()
        personas_seen.add(obs.persona)
        if len(personas_seen) == 3:
            break
    print(f"PASS: personas seen: {personas_seen}")
    assert len(personas_seen) >= 2, f"Should see multiple personas, got: {personas_seen}"


def test_answering_seeded_reset():
    """Seeded resets must produce identical state."""
    env1 = AnsweringEnvironment()
    obs1 = env1.reset(seed=42, domain="hr")

    env2 = AnsweringEnvironment()
    obs2 = env2.reset(seed=42, domain="hr")

    assert obs1.domain == obs2.domain == "hr"
    assert obs1.persona == obs2.persona, f"Persona mismatch: {obs1.persona} vs {obs2.persona}"
    assert obs1.question == obs2.question, "Question mismatch"
    assert obs1.dataset_summary == obs2.dataset_summary
    print(f"PASS: answering seeded reset identical, persona={obs1.persona}")


def test_answering_seeded_reset_different_seeds():
    """Different seeds must produce different state."""
    env1 = AnsweringEnvironment()
    obs1 = env1.reset(seed=42, domain="hr")

    env2 = AnsweringEnvironment()
    obs2 = env2.reset(seed=99, domain="hr")

    assert obs1.persona != obs2.persona or obs1.question != obs2.question, \
        "Different seeds should produce different states"
    print(f"PASS: different seeds produce different states")


def test_answering_client_stores_http_base_url():
    """Client must store _http_base_url for reset_with_seed HTTP calls."""
    url = "https://example.com/answering"
    client = AnsweringEnv(base_url=url)
    assert hasattr(client, "_http_base_url"), "Client missing _http_base_url attribute"
    assert client._http_base_url == url
    assert not hasattr(client, "base_url"), "base_url should not exist on EnvClient"


def test_answering_client_strips_trailing_slash():
    """_http_base_url should strip trailing slashes."""
    client = AnsweringEnv(base_url="https://example.com/answering/")
    assert client._http_base_url == "https://example.com/answering"


if __name__ == "__main__":
    test_answering_reset()
    test_answering_step_good()
    test_answering_step_bad()
    test_answering_all_personas()
    test_answering_seeded_reset()
    test_answering_seeded_reset_different_seeds()
    test_answering_client_stores_http_base_url()
    test_answering_client_strips_trailing_slash()
    print("\nAll answering tests PASSED")
