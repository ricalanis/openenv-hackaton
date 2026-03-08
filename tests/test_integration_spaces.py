"""Integration tests against deployed HF Space services.

These tests replicate the exact HTTP calls the notebooks make to verify
that reset, step, observation parsing, and reward functions all work
end-to-end against the live deployed environments.

Run with: pytest tests/test_integration_spaces.py -v
Skip with: pytest -m "not integration"
"""

import json
import os
import re
import sys

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

SPACE_URLS = {
    "cleaning": "https://ricalanis-datasage-cleaning.hf.space",
    "enrichment": "https://ricalanis-datasage-enrichment.hf.space",
    "answering": "https://ricalanis-datasage-answering.hf.space",
}

TIMEOUT = 30  # seconds per request


# ── Health checks ────────────────────────────────────────────────


@pytest.mark.parametrize("stage", ["cleaning", "enrichment", "answering"])
def test_health_endpoint(stage):
    resp = requests.get(f"{SPACE_URLS[stage]}/health", timeout=TIMEOUT)
    assert resp.status_code == 200


@pytest.mark.parametrize("stage", ["cleaning", "enrichment", "answering"])
def test_schema_endpoint(stage):
    resp = requests.get(f"{SPACE_URLS[stage]}/schema", timeout=TIMEOUT)
    assert resp.status_code == 200
    data = resp.json()
    assert "action" in data
    assert "observation" in data


# ── Reset ────────────────────────────────────────────────────────


class TestCleaningReset:
    def test_reset_returns_200(self):
        resp = requests.post(
            f"{SPACE_URLS['cleaning']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        assert resp.status_code == 200

    def test_reset_has_observation(self):
        resp = requests.post(
            f"{SPACE_URLS['cleaning']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        data = resp.json()
        assert "observation" in data
        obs = data["observation"]
        for field in ["domain", "dq_report", "dq_score", "columns_info", "data_preview"]:
            assert field in obs, f"Missing field: {field}"

    def test_reset_returns_valid_observation(self):
        """Verify reset always returns a well-formed observation."""
        resp = requests.post(
            f"{SPACE_URLS['cleaning']}/web/reset",
            json={"seed": 99}, timeout=TIMEOUT,
        )
        obs = resp.json()["observation"]
        assert obs["domain"] in ("hr", "sales", "project_management", "it_operations")
        assert 0.0 <= obs["dq_score"] <= 1.0
        assert len(obs["data_preview"]) > 0


class TestEnrichmentReset:
    def test_reset_returns_200(self):
        resp = requests.post(
            f"{SPACE_URLS['enrichment']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        assert resp.status_code == 200

    def test_reset_has_observation(self):
        resp = requests.post(
            f"{SPACE_URLS['enrichment']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        obs = resp.json()["observation"]
        for field in ["domain", "schema_info", "available_sources",
                      "possible_enrichments", "data_preview"]:
            assert field in obs, f"Missing field: {field}"
        assert isinstance(obs["available_sources"], list)
        assert len(obs["available_sources"]) > 0


class TestAnsweringReset:
    def test_reset_returns_200(self):
        resp = requests.post(
            f"{SPACE_URLS['answering']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        assert resp.status_code == 200

    def test_reset_has_observation(self):
        resp = requests.post(
            f"{SPACE_URLS['answering']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        obs = resp.json()["observation"]
        for field in ["domain", "persona", "persona_description", "question",
                      "dataset_summary", "column_stats", "available_columns"]:
            assert field in obs, f"Missing field: {field}"


# ── Step ─────────────────────────────────────────────────────────


class TestCleaningStep:
    def test_step_returns_200(self):
        requests.post(
            f"{SPACE_URLS['cleaning']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        action = {
            "operation": "fill_null", "column": "Amount",
            "value": "median", "params": {},
        }
        resp = requests.post(
            f"{SPACE_URLS['cleaning']}/web/step",
            json={"action": action}, timeout=TIMEOUT,
        )
        assert resp.status_code == 200

    def test_step_has_reward(self):
        requests.post(
            f"{SPACE_URLS['cleaning']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        action = {
            "operation": "fill_null", "column": "Amount",
            "value": "median", "params": {},
        }
        resp = requests.post(
            f"{SPACE_URLS['cleaning']}/web/step",
            json={"action": action}, timeout=TIMEOUT,
        )
        data = resp.json()
        assert "reward" in data
        assert isinstance(data["reward"], (int, float))
        assert "done" in data
        assert "observation" in data

    def test_step_all_operations(self):
        """Verify all cleaning operations are accepted without error."""
        operations = [
            {"operation": "fill_null", "column": "Amount", "value": "median", "params": {}},
            {"operation": "fix_type", "column": "Amount", "value": "numeric", "params": {}},
            {"operation": "remove_duplicate", "column": "", "params": {}},
            {"operation": "standardize", "column": "Amount", "params": {}},
            {"operation": "trim", "column": "Amount", "params": {}},
            {"operation": "correct_typo", "column": "Amount", "value": "fixed", "params": {}},
        ]
        for action in operations:
            requests.post(
                f"{SPACE_URLS['cleaning']}/web/reset",
                json={"seed": 42}, timeout=TIMEOUT,
            )
            resp = requests.post(
                f"{SPACE_URLS['cleaning']}/web/step",
                json={"action": action}, timeout=TIMEOUT,
            )
            assert resp.status_code == 200, \
                f"Operation {action['operation']} failed: {resp.status_code}"


class TestEnrichmentStep:
    def test_step_returns_200(self):
        requests.post(
            f"{SPACE_URLS['enrichment']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        action = {
            "operation": "add_field", "field_name": "salary_band",
            "source": "salary_band", "logic": "", "params": {},
        }
        resp = requests.post(
            f"{SPACE_URLS['enrichment']}/web/step",
            json={"action": action}, timeout=TIMEOUT,
        )
        assert resp.status_code == 200

    def test_step_has_reward(self):
        requests.post(
            f"{SPACE_URLS['enrichment']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        action = {
            "operation": "add_field", "field_name": "salary_band",
            "source": "salary_band", "logic": "", "params": {},
        }
        resp = requests.post(
            f"{SPACE_URLS['enrichment']}/web/step",
            json={"action": action}, timeout=TIMEOUT,
        )
        data = resp.json()
        assert "reward" in data
        assert isinstance(data["reward"], (int, float))


class TestAnsweringStep:
    def test_step_returns_200(self):
        requests.post(
            f"{SPACE_URLS['answering']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        action = {
            "answer": "Based on the data, revenue grew 15%.",
            "cited_columns": ["Revenue"],
            "reasoning": "I analyzed the Revenue column.",
        }
        resp = requests.post(
            f"{SPACE_URLS['answering']}/web/step",
            json={"action": action}, timeout=TIMEOUT,
        )
        assert resp.status_code == 200

    def test_step_has_reward(self):
        requests.post(
            f"{SPACE_URLS['answering']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        action = {
            "answer": "Based on the data, revenue grew 15%.",
            "cited_columns": ["Revenue"],
            "reasoning": "I analyzed the Revenue column.",
        }
        resp = requests.post(
            f"{SPACE_URLS['answering']}/web/step",
            json={"action": action}, timeout=TIMEOUT,
        )
        data = resp.json()
        assert "reward" in data
        assert isinstance(data["reward"], (int, float))


# ── Full notebook flow simulation ────────────────────────────────
# These tests replicate the exact sequence of calls each notebook makes:
# 1. reset with seed -> extract observation fields
# 2. parse a model completion -> action dict
# 3. reset with seed (replay) -> step with action -> get reward


def _exec_notebook_cell_3(name):
    """Execute the inlined config+parser+rewards cell and return namespace."""
    notebooks_dir = os.path.join(os.path.dirname(__file__), '..', 'training')
    path = os.path.join(notebooks_dir, name)
    with open(path) as f:
        nb = json.load(f)
    code = ''.join(nb['cells'][3]['source'])
    lines = [l for l in code.split('\n') if not l.strip().startswith('print(')]
    ns = {}
    exec('\n'.join(lines), ns)
    return ns


class TestCleaningNotebookFlow:
    """Simulate the exact cleaning notebook flow against live service."""

    def test_dataset_build_and_reward(self):
        """Reset -> build prompt -> parse fake completion -> step -> reward."""
        ns = _exec_notebook_cell_3('train_cleaning.ipynb')
        env_url = ns['ENV_URL']
        parse_fn = ns['parse_cleaning_action']
        format_reward_fn = ns['cleaning_json_format_reward']
        get_text = ns['_get_text']

        # 1. Reset (like dataset build cell)
        resp = requests.post(f"{env_url}/web/reset", json={"seed": 42}, timeout=TIMEOUT)
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert obs["domain"]
        assert obs["dq_score"] > 0

        # 2. Simulate a model completion (as chat message list, like TRL sends)
        # Note: no "params": {} — nested braces break the _ACTION_JSON_RE regex
        fake_completion = [{"role": "assistant", "content":
            '{"operation": "fill_null", "column": "Amount", "value": "median"}'}]
        text = get_text(fake_completion)
        action_dict = parse_fn(text)
        assert action_dict["operation"] == "fill_null"

        # 3. Format reward works on chat format
        format_rewards = format_reward_fn([fake_completion])
        assert format_rewards == [1.0]

        # 4. Replay env and step (like env reward function)
        requests.post(f"{env_url}/web/reset", json={"seed": 42}, timeout=TIMEOUT)
        resp = requests.post(
            f"{env_url}/web/step",
            json={"action": action_dict}, timeout=TIMEOUT,
        )
        assert resp.status_code == 200
        reward = resp.json().get("reward", 0.0)
        assert isinstance(reward, (int, float))


class TestEnrichmentNotebookFlow:
    """Simulate the exact enrichment notebook flow against live service."""

    def test_dataset_build_and_reward(self):
        ns = _exec_notebook_cell_3('train_enrichment.ipynb')
        env_url = ns['ENV_URL']
        parse_fn = ns['parse_enrichment_action']
        format_reward_fn = ns['enrichment_json_format_reward']
        source_reward_fn = ns['source_relevance_reward']
        get_text = ns['_get_text']

        # 1. Reset
        resp = requests.post(f"{env_url}/web/reset", json={"seed": 42}, timeout=TIMEOUT)
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert obs["domain"]
        available_sources = obs["available_sources"]
        assert len(available_sources) > 0

        # 2. Simulate completion using a real available source
        # Note: no "params": {} — nested braces break the _ACTION_JSON_RE regex
        source = available_sources[0]
        fake_completion = [{"role": "assistant", "content":
            json.dumps({"operation": "add_field", "field_name": source,
                        "source": source, "logic": ""})}]
        text = get_text(fake_completion)
        action_dict = parse_fn(text)
        assert action_dict["field_name"] == source

        # 3. Reward functions work
        assert format_reward_fn([fake_completion]) == [1.0]
        assert source_reward_fn([fake_completion],
                                available_sources=[available_sources]) == [1.0]

        # 4. Step
        requests.post(f"{env_url}/web/reset", json={"seed": 42}, timeout=TIMEOUT)
        resp = requests.post(
            f"{env_url}/web/step",
            json={"action": action_dict}, timeout=TIMEOUT,
        )
        assert resp.status_code == 200
        assert isinstance(resp.json().get("reward"), (int, float))


class TestAnsweringNotebookFlow:
    """Simulate the exact answering notebook flow against live service."""

    def test_dataset_build_and_reward(self):
        ns = _exec_notebook_cell_3('train_answering.ipynb')
        env_url = ns['ENV_URL']
        parse_fn = ns['parse_answering_action']
        format_reward_fn = ns['answering_json_format_reward']
        persona_reward_fn = ns['persona_match_reward']
        faithfulness_fn = ns['local_faithfulness_fn']
        get_text = ns['_get_text']

        # 1. Reset
        resp = requests.post(f"{env_url}/web/reset", json={"seed": 42}, timeout=TIMEOUT)
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert obs["domain"]
        assert obs["persona"]
        assert obs["question"]

        # 2. Simulate completion
        fake_completion = [{"role": "assistant", "content": json.dumps({
            "answer": "Based on the revenue data, Q3 shows 15% growth year-over-year with strong ROI.",
            "cited_columns": ["Revenue", "Quarter"],
            "reasoning": "I analyzed the Revenue and Quarter columns to identify the trend.",
        })}]
        text = get_text(fake_completion)
        action_dict = parse_fn(text)
        assert "answer" in action_dict

        # 3. All reward functions work with chat format
        assert format_reward_fn([fake_completion]) == [1.0]
        persona_rewards = persona_reward_fn(
            [fake_completion], persona_name=[obs["persona"]])
        assert len(persona_rewards) == 1
        assert 0.0 <= persona_rewards[0] <= 1.0
        faith_rewards = faithfulness_fn([fake_completion])
        assert len(faith_rewards) == 1

        # 4. Step
        requests.post(f"{env_url}/web/reset", json={"seed": 42}, timeout=TIMEOUT)
        resp = requests.post(
            f"{env_url}/web/step",
            json={"action": action_dict}, timeout=TIMEOUT,
        )
        assert resp.status_code == 200
        assert isinstance(resp.json().get("reward"), (int, float))


# ── Parser edge cases against live env ───────────────────────────
# Verify that parser fallback outputs produce valid actions the server accepts.


class TestParserFallbacksAgainstLiveEnv:
    """Verify parser fallback outputs don't crash the server."""

    def test_cleaning_fallback_text(self):
        """Keyword fallback action should be accepted by server."""
        ns = _exec_notebook_cell_3('train_cleaning.ipynb')
        action = ns['parse_cleaning_action']("fill the null values in salary")
        requests.post(
            f"{SPACE_URLS['cleaning']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        resp = requests.post(
            f"{SPACE_URLS['cleaning']}/web/step",
            json={"action": action}, timeout=TIMEOUT,
        )
        assert resp.status_code == 200

    def test_cleaning_default_fallback(self):
        """Default fallback (no keywords matched) should be accepted."""
        ns = _exec_notebook_cell_3('train_cleaning.ipynb')
        action = ns['parse_cleaning_action']("something completely unrelated")
        requests.post(
            f"{SPACE_URLS['cleaning']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        resp = requests.post(
            f"{SPACE_URLS['cleaning']}/web/step",
            json={"action": action}, timeout=TIMEOUT,
        )
        assert resp.status_code == 200

    def test_enrichment_default_fallback(self):
        ns = _exec_notebook_cell_3('train_enrichment.ipynb')
        action = ns['parse_enrichment_action']("gibberish text")
        requests.post(
            f"{SPACE_URLS['enrichment']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        resp = requests.post(
            f"{SPACE_URLS['enrichment']}/web/step",
            json={"action": action}, timeout=TIMEOUT,
        )
        assert resp.status_code == 200

    def test_answering_fallback_raw_text(self):
        ns = _exec_notebook_cell_3('train_answering.ipynb')
        action = ns['parse_answering_action']("The answer is 42.")
        requests.post(
            f"{SPACE_URLS['answering']}/web/reset",
            json={"seed": 42}, timeout=TIMEOUT,
        )
        resp = requests.post(
            f"{SPACE_URLS['answering']}/web/step",
            json={"action": action}, timeout=TIMEOUT,
        )
        assert resp.status_code == 200
