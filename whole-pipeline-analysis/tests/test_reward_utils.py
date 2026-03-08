"""Tests for simplified reward_utils."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.shared.reward_utils import cleaning_reward, enrichment_reward, answering_reward


def test_cleaning_reward_uses_delta():
    r1 = cleaning_reward(dq_before=0.50, dq_after=0.80)
    r2 = cleaning_reward(dq_before=0.85, dq_after=0.87)
    r3 = cleaning_reward(dq_before=0.70, dq_after=0.70)
    assert r1 > r3, "Big improvement should beat no improvement"
    assert r3 >= 0, "No improvement should still be non-negative"
    print(f"PASS: cleaning_reward r1={r1:.4f} r2={r2:.4f} r3={r3:.4f}")


def test_cleaning_reward_no_downstream_cache():
    r1 = cleaning_reward(dq_before=0.30, dq_after=0.80)
    r2 = cleaning_reward(dq_before=0.75, dq_after=0.80)
    assert r1 > r2, "Bigger improvement should give bigger reward"
    print(f"PASS: no downstream cache, r1={r1:.4f} > r2={r2:.4f}")


def test_enrichment_reward_direct():
    assert enrichment_reward(0.0) == 0.0
    assert enrichment_reward(1.0) == 1.0
    assert enrichment_reward(0.5) == 0.5
    print("PASS: enrichment_reward is direct coverage")


def test_answering_reward_with_patronus():
    r = answering_reward(0.5, 0.8, patronus_score=0.9)
    expected = 0.40 * 0.9 + 0.60 * 0.8
    assert abs(r - expected) < 0.001
    print(f"PASS: answering_reward with patronus = {r:.4f}")


def test_answering_reward_without_patronus():
    r = answering_reward(0.5, 0.8, patronus_score=None)
    expected = 0.30 * 0.5 + 0.70 * 0.8
    assert abs(r - expected) < 0.001
    print(f"PASS: answering_reward without patronus = {r:.4f}")


if __name__ == "__main__":
    test_cleaning_reward_uses_delta()
    test_cleaning_reward_no_downstream_cache()
    test_enrichment_reward_direct()
    test_answering_reward_with_patronus()
    test_answering_reward_without_patronus()
    print("\nAll reward_utils tests PASSED")
