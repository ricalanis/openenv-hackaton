"""Smoke tests for the enrichment environment."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environments', 'enrichment'))

from environments.enrichment.server.enrichment_environment import EnrichmentEnvironment
from environments.enrichment.models import EnrichmentAction


def test_enrichment_reset():
    env = EnrichmentEnvironment()
    obs = env.reset()
    assert obs.domain in ["hr", "sales", "pm", "it_ops"]
    assert obs.enrichment_coverage == 0.0
    assert len(obs.available_sources) > 0
    assert len(obs.possible_enrichments) > 0
    assert obs.done is False
    print(f"PASS: reset() domain={obs.domain} sources={obs.available_sources}")
    return obs


def test_enrichment_step():
    env = EnrichmentEnvironment()
    obs = env.reset()

    # Add first available enrichment
    source = obs.available_sources[0]
    action = EnrichmentAction(
        operation="add_field",
        field_name=source,
        source=source,
    )
    obs2 = env.step(action)
    assert obs2.enrichment_coverage > 0
    assert source in obs2.fields_added
    assert obs2.reward >= 0
    print(f"PASS: step({source}) coverage={obs2.enrichment_coverage:.2f} reward={obs2.reward:.4f}")
    return obs2


def test_enrichment_full_episode():
    env = EnrichmentEnvironment()
    obs = env.reset()
    domain = obs.domain
    step = 0

    for source in obs.available_sources:
        action = EnrichmentAction(
            operation="add_field",
            field_name=source,
            source=source,
        )
        obs = env.step(action)
        step += 1
        print(f"  step={step} added={source} coverage={obs.enrichment_coverage:.2f} done={obs.done}")
        if obs.done:
            break

    assert obs.enrichment_coverage > 0.5, f"Coverage should be >50%: {obs.enrichment_coverage}"
    print(f"PASS: full episode domain={domain} coverage={obs.enrichment_coverage:.2f} steps={step}")


if __name__ == "__main__":
    test_enrichment_reset()
    test_enrichment_step()
    test_enrichment_full_episode()
    print("\nAll enrichment tests PASSED")
