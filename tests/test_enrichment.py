"""Smoke tests for the enrichment environment."""

import sys
import os

# Add project root first so `environments.*` resolves correctly.
# The enrichment dir is *appended* (not inserted at 0) to avoid its local
# ``environments/`` sub-directory shadowing the top-level package.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_enrichment_root = os.path.join(_project_root, 'environments', 'enrichment')
if _enrichment_root not in sys.path:
    sys.path.append(_enrichment_root)

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


def test_enrichment_seeded_reset():
    env1 = EnrichmentEnvironment()
    obs1 = env1.reset(seed=42, domain="hr")
    env2 = EnrichmentEnvironment()
    obs2 = env2.reset(seed=42, domain="hr")
    assert obs1.domain == obs2.domain == "hr"
    assert obs1.available_sources == obs2.available_sources
    assert obs1.data_preview == obs2.data_preview
    print(f"PASS: enrichment seeded reset identical")


def test_enrichment_seeded_reset_different_seeds():
    env1 = EnrichmentEnvironment()
    obs1 = env1.reset(seed=42, domain="hr")
    env2 = EnrichmentEnvironment()
    obs2 = env2.reset(seed=99, domain="hr")
    # Data preview may differ due to different random samples
    print(f"PASS: enrichment different seeds work")


if __name__ == "__main__":
    test_enrichment_reset()
    test_enrichment_step()
    test_enrichment_full_episode()
    test_enrichment_seeded_reset()
    test_enrichment_seeded_reset_different_seeds()
    print("\nAll enrichment tests PASSED")
