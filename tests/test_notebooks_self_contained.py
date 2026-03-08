"""Tests that training notebooks are fully self-contained for Colab.

Validates:
- No project imports (training.*, environments.*)
- No repo clone / sys.path manipulation
- No openenv-core dependency
- Correct cell structure (11 cells)
- Correct HF Space URLs
- requests.post used for env interaction
- Inlined code produces identical results to shared modules
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'training')

NOTEBOOK_NAMES = [
    'train_cleaning.ipynb',
    'train_enrichment.ipynb',
    'train_answering.ipynb',
]

EXPECTED_URLS = {
    'train_cleaning.ipynb': 'https://ricalanis-datasage-cleaning.hf.space',
    'train_enrichment.ipynb': 'https://ricalanis-datasage-enrichment.hf.space',
    'train_answering.ipynb': 'https://ricalanis-datasage-answering.hf.space',
}

EXPECTED_HF_REPOS = {
    'train_cleaning.ipynb': 'ricalanis/datasage-qwen-cleaning',
    'train_enrichment.ipynb': 'ricalanis/datasage-qwen-enrichment',
    'train_answering.ipynb': 'ricalanis/datasage-qwen-answering',
}


def _load_notebook(name):
    path = os.path.join(NOTEBOOKS_DIR, name)
    with open(path) as f:
        return json.load(f)


def _get_all_source(nb):
    """Concatenate all code cell sources into one string."""
    parts = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            parts.append(''.join(cell['source']))
    return '\n'.join(parts)


# ── Structure tests ───────────────────────────────────────────────


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_notebook_has_11_cells(name):
    nb = _load_notebook(name)
    assert len(nb['cells']) == 11, f"{name} has {len(nb['cells'])} cells, expected 11"


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_first_cell_is_markdown(name):
    nb = _load_notebook(name)
    assert nb['cells'][0]['cell_type'] == 'markdown'


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_colab_badge_present(name):
    nb = _load_notebook(name)
    md = ''.join(nb['cells'][0]['source'])
    assert 'colab.research.google.com' in md


# ── No project imports ────────────────────────────────────────────


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_no_training_imports(name):
    src = _get_all_source(_load_notebook(name))
    assert 'from training.' not in src, f"{name} still imports from training.*"
    assert 'import training.' not in src, f"{name} still imports training.*"


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_no_environments_imports(name):
    src = _get_all_source(_load_notebook(name))
    assert 'from environments.' not in src, f"{name} still imports from environments.*"
    assert 'import environments.' not in src, f"{name} still imports environments.*"


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_no_openenv_core(name):
    src = _get_all_source(_load_notebook(name))
    assert 'openenv-core' not in src, f"{name} still references openenv-core"


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_no_repo_clone(name):
    src = _get_all_source(_load_notebook(name))
    assert 'git clone' not in src, f"{name} still clones the repo"


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_no_sys_path_manipulation(name):
    src = _get_all_source(_load_notebook(name))
    assert 'sys.path' not in src, f"{name} still manipulates sys.path"


# ── Correct URLs and config ──────────────────────────────────────


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_correct_env_url(name):
    src = _get_all_source(_load_notebook(name))
    expected = EXPECTED_URLS[name]
    assert expected in src, f"{name} missing ENV_URL {expected}"


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_correct_hf_repo(name):
    src = _get_all_source(_load_notebook(name))
    expected = EXPECTED_HF_REPOS[name]
    assert expected in src, f"{name} missing HF_REPO {expected}"


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_uses_requests_post(name):
    src = _get_all_source(_load_notebook(name))
    assert 'requests.post' in src, f"{name} doesn't use requests.post"


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_pip_install_has_requests(name):
    nb = _load_notebook(name)
    pip_cell = ''.join(nb['cells'][1]['source'])
    assert 'requests' in pip_cell, f"{name} pip install missing requests"


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_no_use_vllm(name):
    """vLLM colocate mode crashes with PEFT/LoRA models in Colab."""
    src = _get_all_source(_load_notebook(name))
    assert 'use_vllm' not in src, f"{name} still has use_vllm (incompatible with PEFT in Colab)"


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_has_get_text_helper(name):
    """All notebooks must have _get_text to handle chat message completions."""
    src = _get_all_source(_load_notebook(name))
    assert '_get_text' in src, f"{name} missing _get_text helper"


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_step_payload_not_wrapped(name):
    """Step calls must send action fields at top level, not nested in {"action": ...}."""
    src = _get_all_source(_load_notebook(name))
    assert 'json={"action"' not in src, \
        f"{name} wraps step payload in {{\"action\": ...}} — server expects top-level fields"


@pytest.mark.parametrize('name', NOTEBOOK_NAMES)
def test_pip_install_no_pydantic(name):
    nb = _load_notebook(name)
    pip_cell = ''.join(nb['cells'][1]['source'])
    assert 'pydantic' not in pip_cell, f"{name} pip install still has pydantic"


# ── Inlined code parity with shared modules ──────────────────────

# Execute the inlined config+parser+reward cell to get the functions,
# then compare outputs with the original shared modules.


def _exec_cell_3(name):
    """Execute cell 3 (config+parser+rewards) and return its namespace."""
    nb = _load_notebook(name)
    code = ''.join(nb['cells'][3]['source'])
    # Remove print statements that reference undefined vars at exec time
    lines = [l for l in code.split('\n') if not l.strip().startswith('print(')]
    ns = {}
    exec('\n'.join(lines), ns)
    return ns


# ── _get_text handles both str and chat message formats ──────────


class TestGetTextHelper:
    """Verify _get_text extracts text from both str and chat message formats."""

    @pytest.mark.parametrize('name', NOTEBOOK_NAMES)
    def test_get_text_with_string(self, name):
        ns = _exec_cell_3(name)
        assert ns['_get_text']("hello") == "hello"

    @pytest.mark.parametrize('name', NOTEBOOK_NAMES)
    def test_get_text_with_chat_messages(self, name):
        ns = _exec_cell_3(name)
        msgs = [{"role": "assistant", "content": "the answer"}]
        assert ns['_get_text'](msgs) == "the answer"

    @pytest.mark.parametrize('name', NOTEBOOK_NAMES)
    def test_get_text_with_empty_list(self, name):
        ns = _exec_cell_3(name)
        assert ns['_get_text']([]) == ""


class TestRewardWithChatFormat:
    """Verify reward functions work with chat message completions (TRL format)."""

    def test_cleaning_json_format_chat(self):
        ns = _exec_cell_3('train_cleaning.ipynb')
        fn = ns['cleaning_json_format_reward']
        completions = [
            [{"role": "assistant", "content": '{"operation": "fill_null", "column": "age", "value": "median"}'}],
            [{"role": "assistant", "content": "plain text"}],
        ]
        assert fn(completions) == [1.0, 0.0]

    def test_enrichment_json_format_chat(self):
        ns = _exec_cell_3('train_enrichment.ipynb')
        fn = ns['enrichment_json_format_reward']
        completions = [
            [{"role": "assistant", "content": '{"operation": "add_field", "field_name": "salary_band", "source": "x"}'}],
            [{"role": "assistant", "content": "plain text"}],
        ]
        assert fn(completions) == [1.0, 0.0]

    def test_answering_json_format_chat(self):
        ns = _exec_cell_3('train_answering.ipynb')
        fn = ns['answering_json_format_reward']
        completions = [
            [{"role": "assistant", "content": '{"answer": "yes", "cited_columns": ["c"], "reasoning": "r"}'}],
            [{"role": "assistant", "content": "plain text"}],
        ]
        assert fn(completions) == [1.0, 0.0]

    def test_persona_match_chat(self):
        ns = _exec_cell_3('train_answering.ipynb')
        fn = ns['persona_match_reward']
        completions = [
            [{"role": "assistant", "content": "The revenue trend shows 15% growth year-over-year with strong ROI impact on budget margins."}],
        ]
        rewards = fn(completions, persona_name=["Executive"])
        assert rewards[0] > 0.3

    def test_source_relevance_chat(self):
        ns = _exec_cell_3('train_enrichment.ipynb')
        fn = ns['source_relevance_reward']
        completions = [
            [{"role": "assistant", "content": '{"operation": "add_field", "field_name": "salary_band", "source": "salary_band"}'}],
        ]
        rewards = fn(completions, available_sources=[["salary_band", "tenure_risk"]])
        assert rewards == [1.0]


class TestCleaningParity:
    """Verify inlined cleaning code matches shared modules."""

    def test_parse_cleaning_json(self):
        from training.shared.parsers import parse_cleaning_action as original
        ns = _exec_cell_3('train_cleaning.ipynb')
        inlined = ns['parse_cleaning_action']

        text = '{"operation": "fill_null", "column": "age", "value": "median"}'
        assert inlined(text) == original(text)

    def test_parse_cleaning_fallback_fill(self):
        from training.shared.parsers import parse_cleaning_action as original
        ns = _exec_cell_3('train_cleaning.ipynb')
        inlined = ns['parse_cleaning_action']

        text = 'We should fill the null values in "salary" column'
        assert inlined(text) == original(text)

    def test_parse_cleaning_fallback_duplicate(self):
        from training.shared.parsers import parse_cleaning_action as original
        ns = _exec_cell_3('train_cleaning.ipynb')
        inlined = ns['parse_cleaning_action']

        text = 'Remove duplicate rows from the dataset'
        assert inlined(text) == original(text)

    def test_parse_cleaning_default(self):
        from training.shared.parsers import parse_cleaning_action as original
        ns = _exec_cell_3('train_cleaning.ipynb')
        inlined = ns['parse_cleaning_action']

        text = 'something unrelated'
        assert inlined(text) == original(text)

    def test_cleaning_json_format_reward(self):
        from training.shared.rewards import cleaning_json_format_reward as original
        ns = _exec_cell_3('train_cleaning.ipynb')
        inlined = ns['cleaning_json_format_reward']

        cases = [
            '{"operation": "fill_null", "column": "age", "value": "median"}',
            '{"operation": "fill_null"}',
            '{"operation": "nuke_it", "column": "age"}',
            'plain text with no json',
        ]
        assert inlined(cases) == original(cases)


class TestEnrichmentParity:
    """Verify inlined enrichment code matches shared modules."""

    def test_parse_enrichment_json(self):
        from training.shared.parsers import parse_enrichment_action as original
        ns = _exec_cell_3('train_enrichment.ipynb')
        inlined = ns['parse_enrichment_action']

        text = '{"operation": "add_field", "field_name": "salary_band", "source": "salary_band"}'
        assert inlined(text) == original(text)

    def test_parse_enrichment_keyword(self):
        from training.shared.parsers import parse_enrichment_action as original
        ns = _exec_cell_3('train_enrichment.ipynb')
        inlined = ns['parse_enrichment_action']

        text = 'Add the salary band enrichment to this dataset'
        assert inlined(text) == original(text)

    def test_parse_enrichment_default(self):
        from training.shared.parsers import parse_enrichment_action as original
        ns = _exec_cell_3('train_enrichment.ipynb')
        inlined = ns['parse_enrichment_action']

        text = 'something unrelated'
        assert inlined(text) == original(text)

    def test_source_relevance_reward(self):
        from training.shared.rewards import source_relevance_reward as original
        ns = _exec_cell_3('train_enrichment.ipynb')
        inlined = ns['source_relevance_reward']

        completions = [
            '{"operation": "add_field", "field_name": "salary_band", "source": "salary_band"}',
            'just text with no JSON',
        ]
        kwargs = {"available_sources": [["salary_band", "tenure_risk"], ["salary_band"]]}
        assert inlined(completions, **kwargs) == original(completions, **kwargs)

    def test_enrichment_json_format_reward(self):
        from training.shared.rewards import enrichment_json_format_reward as original
        ns = _exec_cell_3('train_enrichment.ipynb')
        inlined = ns['enrichment_json_format_reward']

        cases = [
            '{"operation": "add_field", "field_name": "salary_band", "source": "x"}',
            '{"operation": "add_field", "field_name": "unknown"}',
            'plain text',
        ]
        assert inlined(cases) == original(cases)


class TestAnsweringParity:
    """Verify inlined answering code matches shared modules."""

    def test_parse_answering_json(self):
        from training.shared.parsers import parse_answering_action as original
        ns = _exec_cell_3('train_answering.ipynb')
        inlined = ns['parse_answering_action']

        text = '{"answer": "The data shows 15% growth", "cited_columns": ["Revenue"], "reasoning": "Based on..."}'
        assert inlined(text) == original(text)

    def test_parse_answering_fallback(self):
        from training.shared.parsers import parse_answering_action as original
        ns = _exec_cell_3('train_answering.ipynb')
        inlined = ns['parse_answering_action']

        text = 'The revenue grew by 15% this quarter'
        assert inlined(text) == original(text)

    def test_persona_match_reward(self):
        from training.shared.rewards import persona_match_reward as original
        ns = _exec_cell_3('train_answering.ipynb')
        inlined = ns['persona_match_reward']

        completions = [
            "The revenue trend shows 15% growth year-over-year with strong ROI impact on budget margins.",
            "Your next task is to fix the login page bug by the deadline.",
        ]
        kwargs = {"persona_name": ["Executive", "Individual Contributor"]}
        assert inlined(completions, **kwargs) == original(completions, **kwargs)

    def test_local_faithfulness_fn(self):
        from training.shared.rewards import local_faithfulness_fn as original
        ns = _exec_cell_3('train_answering.ipynb')
        inlined = ns['local_faithfulness_fn']

        cases = [
            "Based on EmployeeCount and SalaryRange, the team has 250 members with 85% retention.",
            "idk",
            "I believe this might be correct, probably",
        ]
        assert inlined(cases) == original(cases)

    def test_answering_json_format_reward(self):
        from training.shared.rewards import answering_json_format_reward as original
        ns = _exec_cell_3('train_answering.ipynb')
        inlined = ns['answering_json_format_reward']

        cases = [
            '{"answer": "The data shows...", "cited_columns": ["col1"], "reasoning": "I analyzed..."}',
            '{"answer": "The data shows...", "cited_columns": ["col1"]}',
            '{"answer": "The data shows..."}',
            'plain text',
        ]
        assert inlined(cases) == original(cases)

    def test_score_persona_alignment(self):
        from environments.shared.personas import score_persona_alignment as original
        from environments.shared.personas import PERSONAS as original_personas
        ns = _exec_cell_3('train_answering.ipynb')
        inlined_score = ns['score_persona_alignment']
        inlined_personas = ns['PERSONAS']

        text = "The revenue trend shows 15% growth with strong ROI and budget impact."
        # Compare for each persona
        for orig_p, inl_p in zip(original_personas, inlined_personas):
            assert orig_p.name == inl_p.name
            assert original(text, orig_p) == inlined_score(text, inl_p)
