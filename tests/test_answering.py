"""Smoke tests for the answering environment."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environments', 'answering'))

from environments.answering.server.answering_environment import AnsweringEnvironment
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


if __name__ == "__main__":
    test_answering_reset()
    test_answering_step_good()
    test_answering_step_bad()
    test_answering_all_personas()
    print("\nAll answering tests PASSED")
