# Reward Pipeline Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the fundamental context mismatch in GRPO training by using `rollout_func` so models see real environment observations and rewards reflect actual action quality.

**Architecture:** Replace static prompts + mismatched env rewards with `rollout_func` + `generate_rollout_completions`. Each rollout: (1) resets env with seed, (2) injects observation into prompt, (3) generates completions, (4) evaluates each completion against the same seeded env state. Extra fields flow from rollout to reward functions via kwargs.

**Tech Stack:** TRL (GRPOTrainer, rollout_func, generate_rollout_completions), OpenEnv HTTP clients, Unsloth + vLLM colocate, FastAPI.

---

### Task 1: Add Seeded Reset to Cleaning Environment

**Files:**
- Modify: `environments/cleaning/server/cleaning_environment.py:63-91`
- Modify: `environments/cleaning/server/app.py:26-32`
- Modify: `environments/cleaning/client.py:12-63`
- Test: `tests/test_cleaning.py`

**Step 1: Write the failing test**

Add to `tests/test_cleaning.py`:

```python
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

    # Very unlikely to be identical with different seeds
    assert obs1.dq_score != obs2.dq_score or obs1.data_preview != obs2.data_preview, \
        "Different seeds should produce different states"
    print(f"PASS: different seeds produce different states")
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ricalanis/Documents/dev/openenv-hackaton && python -m pytest tests/test_cleaning.py::test_cleaning_seeded_reset -v`
Expected: FAIL with `TypeError: reset() got unexpected keyword argument 'seed'`

**Step 3: Implement seeded reset on CleaningEnvironment**

Modify `environments/cleaning/server/cleaning_environment.py`. Change `reset(self)` to:

```python
def reset(self, seed: int | None = None, domain: str | None = None) -> CleaningObservation:
    """Pick a domain, load a 50-row batch, inject corruption.

    Args:
        seed: Optional RNG seed for reproducible state.
        domain: Optional domain override (hr, sales, pm, it_ops).
    """
    self._state = State(episode_id=str(uuid4()), step_count=0)

    # Set seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if domain and domain in DOMAINS:
        self._domain_name = domain
    else:
        self._domain_name = random.choice(list(DOMAINS.keys()))
    self._domain_config = DOMAINS[self._domain_name]

    # Load raw data and sample 50 rows
    raw_df = load_domain_data(self._domain_name, sample_size=50)
    self._df = inject_corruption(raw_df, self._domain_config, rate=0.15)

    dq = compute_dq_score(self._df, self._domain_config)
    dq_report = (
        f"completeness={dq['completeness']:.4f}  "
        f"consistency={dq['consistency']:.4f}  "
        f"uniqueness={dq['uniqueness']:.4f}"
    )

    return CleaningObservation(
        domain=self._domain_name,
        data_preview=format_preview(self._df),
        dq_report=dq_report,
        dq_score=dq["overall"],
        columns_info=format_columns_info(self._df, self._domain_config),
        step_number=0,
        max_steps=self._max_steps,
        done=False,
        reward=0.0,
    )
```

**Step 4: Add `/reset-with-seed` endpoint to app.py**

Modify `environments/cleaning/server/app.py`. After the `app = create_app(...)` block, add:

```python
from fastapi import Request

@app.post("/reset-with-seed")
async def reset_with_seed(request: Request):
    """Reset environment with a specific seed for reproducible state."""
    body = await request.json()
    seed = body.get("seed")
    domain = body.get("domain")
    env = CleaningEnvironment()
    obs = env.reset(seed=seed, domain=domain)
    return {
        "observation": obs.model_dump(),
        "done": obs.done,
        "reward": obs.reward,
    }
```

**Step 5: Add `reset_with_seed` method to client.py**

Add to `environments/cleaning/client.py`:

```python
import requests

class CleaningEnv(EnvClient[CleaningAction, CleaningObservation, State]):
    # ... existing code ...

    def reset_with_seed(self, seed: int, domain: str | None = None) -> StepResult[CleaningObservation]:
        """Reset with a specific seed for reproducible state."""
        payload = {"seed": seed}
        if domain:
            payload["domain"] = domain
        resp = requests.post(f"{self.base_url}/reset-with-seed", json=payload)
        resp.raise_for_status()
        return self._parse_result(resp.json())
```

**Step 6: Run tests to verify they pass**

Run: `cd /Users/ricalanis/Documents/dev/openenv-hackaton && python -m pytest tests/test_cleaning.py -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add environments/cleaning/server/cleaning_environment.py environments/cleaning/server/app.py environments/cleaning/client.py tests/test_cleaning.py
git commit -m "feat(cleaning): add seeded reset for reproducible GRPO training"
```

---

### Task 2: Add Seeded Reset to Enrichment Environment

**Files:**
- Modify: `environments/enrichment/server/enrichment_environment.py:56-106`
- Modify: `environments/enrichment/server/app.py:25-31`
- Modify: `environments/enrichment/client.py:12-74`
- Test: `tests/test_enrichment.py`

**Step 1: Write the failing test**

Add to `tests/test_enrichment.py`:

```python
def test_enrichment_seeded_reset():
    """Seeded resets must produce identical state."""
    env1 = EnrichmentEnvironment()
    obs1 = env1.reset(seed=42, domain="hr")

    env2 = EnrichmentEnvironment()
    obs2 = env2.reset(seed=42, domain="hr")

    assert obs1.domain == obs2.domain == "hr"
    assert obs1.available_sources == obs2.available_sources
    assert obs1.data_preview == obs2.data_preview
    print(f"PASS: enrichment seeded reset identical, sources={obs1.available_sources}")
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ricalanis/Documents/dev/openenv-hackaton && python -m pytest tests/test_enrichment.py::test_enrichment_seeded_reset -v`
Expected: FAIL with `TypeError`

**Step 3: Implement seeded reset on EnrichmentEnvironment**

Modify `environments/enrichment/server/enrichment_environment.py`. Change `reset(self)` to:

```python
def reset(self, seed: int | None = None, domain: str | None = None) -> EnrichmentObservation:
    """Reset the environment.

    Args:
        seed: Optional RNG seed for reproducible state.
        domain: Optional domain override.
    """
    self._state = State(episode_id=str(uuid4()), step_count=0)
    self._fields_added = []

    if seed is not None:
        random.seed(seed)

    if domain and domain in DOMAINS:
        self._domain = domain
    else:
        self._domain = random.choice(list(DOMAINS.keys()))
    self._domain_config = DOMAINS[self._domain]

    self._df = load_domain_data(self._domain, sample_size=50)

    # ... rest of method unchanged (build schema_info, available sources, return obs) ...
```

Note: Add `import numpy as np` at the top if not present, and add `np.random.seed(seed)` after `random.seed(seed)`.

**Step 4: Add `/reset-with-seed` to enrichment app.py**

Same pattern as Task 1. Add after `app = create_app(...)`:

```python
from fastapi import Request
from .enrichment_environment import EnrichmentEnvironment

@app.post("/reset-with-seed")
async def reset_with_seed(request: Request):
    body = await request.json()
    env = EnrichmentEnvironment()
    obs = env.reset(seed=body.get("seed"), domain=body.get("domain"))
    return {"observation": obs.model_dump(), "done": obs.done, "reward": obs.reward}
```

**Step 5: Add `reset_with_seed` to enrichment client.py**

Same pattern as Task 1. Add `reset_with_seed(self, seed, domain=None)` method.

**Step 6: Run tests**

Run: `cd /Users/ricalanis/Documents/dev/openenv-hackaton && python -m pytest tests/test_enrichment.py -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add environments/enrichment/ tests/test_enrichment.py
git commit -m "feat(enrichment): add seeded reset for reproducible GRPO training"
```

---

### Task 3: Add Seeded Reset to Answering Environment

**Files:**
- Modify: `environments/answering/server/answering_environment.py:60-114`
- Modify: `environments/answering/server/app.py:39-45`
- Modify: `environments/answering/client.py:12-94`
- Test: `tests/test_answering.py`

**Step 1: Write the failing test**

Add to `tests/test_answering.py`:

```python
def test_answering_seeded_reset():
    """Seeded resets must produce identical state (same persona, question, data)."""
    env1 = AnsweringEnvironment()
    obs1 = env1.reset(seed=42, domain="hr")

    env2 = AnsweringEnvironment()
    obs2 = env2.reset(seed=42, domain="hr")

    assert obs1.domain == obs2.domain == "hr"
    assert obs1.persona == obs2.persona, f"Persona mismatch: {obs1.persona} vs {obs2.persona}"
    assert obs1.question == obs2.question, f"Question mismatch"
    assert obs1.dataset_summary == obs2.dataset_summary
    print(f"PASS: answering seeded reset identical, persona={obs1.persona}, q={obs1.question[:50]}")
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ricalanis/Documents/dev/openenv-hackaton && python -m pytest tests/test_answering.py::test_answering_seeded_reset -v`
Expected: FAIL with `TypeError`

**Step 3: Implement seeded reset on AnsweringEnvironment**

Modify `environments/answering/server/answering_environment.py`. Change `reset(self)` to:

```python
def reset(self, seed: int | None = None, domain: str | None = None) -> AnsweringObservation:
    """Pick a domain, persona, and question; load enriched data summary.

    Args:
        seed: Optional RNG seed for reproducible state.
        domain: Optional domain override.
    """
    self._state = State(episode_id=str(uuid4()), step_count=0)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if domain and domain in DOMAINS:
        self._domain_name = domain
    else:
        self._domain_name = random.choice(list(DOMAINS.keys()))
    self._domain_config = DOMAINS[self._domain_name]

    self._persona = random.choice(PERSONAS)
    self._question = random.choice(self._domain_config.example_questions)

    # ... rest unchanged (load data, compute summary, return obs) ...
```

**Step 4: Add `/reset-with-seed` to answering app.py**

Same pattern as Tasks 1-2.

**Step 5: Add `reset_with_seed` to answering client.py**

Same pattern.

**Step 6: Run tests**

Run: `cd /Users/ricalanis/Documents/dev/openenv-hackaton && python -m pytest tests/test_answering.py -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add environments/answering/ tests/test_answering.py
git commit -m "feat(answering): add seeded reset for reproducible GRPO training"
```

---

### Task 4: Simplify reward_utils.py

**Files:**
- Modify: `environments/shared/reward_utils.py`
- Modify: `environments/cleaning/server/cleaning_environment.py:125` (update call)
- Modify: `environments/enrichment/server/enrichment_environment.py:159` (update call)
- Test: `tests/test_reward_utils.py` (new)

**Step 1: Write the failing test**

Create `tests/test_reward_utils.py`:

```python
"""Tests for simplified reward_utils."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.shared.reward_utils import cleaning_reward, enrichment_reward, answering_reward


def test_cleaning_reward_uses_delta():
    """Cleaning reward should value improvement, not just absolute DQ."""
    # Big improvement from low base
    r1 = cleaning_reward(dq_before=0.50, dq_after=0.80)
    # Small improvement from high base
    r2 = cleaning_reward(dq_before=0.85, dq_after=0.87)
    # No improvement
    r3 = cleaning_reward(dq_before=0.70, dq_after=0.70)

    assert r1 > r3, "Big improvement should beat no improvement"
    assert r3 >= 0, "No improvement should still be non-negative"
    print(f"PASS: cleaning_reward r1={r1:.4f} r2={r2:.4f} r3={r3:.4f}")


def test_cleaning_reward_no_downstream_cache():
    """cleaning_reward should NOT use DOWNSTREAM_CACHE."""
    # Same dq_after, different dq_before -> different reward
    r1 = cleaning_reward(dq_before=0.30, dq_after=0.80)
    r2 = cleaning_reward(dq_before=0.75, dq_after=0.80)
    assert r1 > r2, "Bigger improvement should give bigger reward"
    print(f"PASS: no downstream cache, r1={r1:.4f} > r2={r2:.4f}")


def test_enrichment_reward_direct():
    """Enrichment reward should be direct coverage score."""
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ricalanis/Documents/dev/openenv-hackaton && python -m pytest tests/test_reward_utils.py -v`
Expected: FAIL with `TypeError: cleaning_reward() got unexpected keyword argument 'dq_before'`

**Step 3: Rewrite reward_utils.py**

Replace `environments/shared/reward_utils.py` entirely:

```python
"""Reward computation for DataSage training pipeline."""


def cleaning_reward(dq_before: float, dq_after: float) -> float:
    """Compute cleaning stage reward.

    Rewards both absolute DQ quality and the improvement delta.
    50% absolute quality + 50% improvement (scaled up to be meaningful).
    """
    improvement = max(0.0, dq_after - dq_before)
    return round(0.50 * dq_after + 0.50 * min(improvement * 5.0, 1.0), 4)


def enrichment_reward(coverage: float) -> float:
    """Compute enrichment stage reward.

    Direct coverage signal — no downstream mixing.
    """
    return round(coverage, 4)


def answering_reward(faithfulness: float, persona_relevance: float,
                     patronus_score: float | None = None) -> float:
    """Compute answering stage reward.

    Without Patronus: 0.30 * faithfulness + 0.70 * persona_relevance
    With Patronus:    0.40 * patronus_score + 0.60 * persona_relevance
    """
    if patronus_score is not None:
        return round(0.40 * patronus_score + 0.60 * persona_relevance, 4)
    return round(0.30 * faithfulness + 0.70 * persona_relevance, 4)
```

**Step 4: Update cleaning_environment.py call site**

In `environments/cleaning/server/cleaning_environment.py:124-125`, change:

```python
# BEFORE:
reward = cleaning_reward(dq["overall"])

# AFTER:
dq_before = self._initial_dq  # Need to store this in reset()
reward = cleaning_reward(dq_before, dq["overall"])
```

Also add `self._initial_dq = dq["overall"]` at the end of the `reset()` method (before the return).

**Step 5: Update enrichment_environment.py call site**

In `environments/enrichment/server/enrichment_environment.py:149-159`, change:

```python
# BEFORE:
if coverage > 0.80:
    downstream_bucket = "excellent"
elif coverage > 0.50:
    downstream_bucket = "good"
elif coverage > 0.30:
    downstream_bucket = "fair"
else:
    downstream_bucket = "poor"
reward = enrichment_reward(coverage, downstream_bucket)

# AFTER:
reward = enrichment_reward(coverage)
```

Delete the entire downstream_bucket block (lines 149-159 approximately).

**Step 6: Run all tests**

Run: `cd /Users/ricalanis/Documents/dev/openenv-hackaton && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add environments/shared/reward_utils.py environments/cleaning/server/cleaning_environment.py environments/enrichment/server/enrichment_environment.py tests/test_reward_utils.py
git commit -m "refactor(rewards): remove redundant downstream cache, add improvement delta"
```

---

### Task 5: Rewrite train_cleaning.py with rollout_func

**Files:**
- Modify: `training/train_cleaning.py` (major rewrite)
- Test: Run training with `--dry-run` flag or small subset

This is the core fix. The training script changes from "static prompts + mismatched env reward" to "rollout_func that injects env observation into prompt and evaluates against same state."

**Step 1: Create the new training script**

Replace `training/train_cleaning.py`. Key structural changes:

1. **Remove static TASK_PROMPTS** — prompts are now built dynamically from env observations
2. **Add `cleaning_rollout` function** — the rollout_func that controls generation
3. **Add prompt builder** — constructs prompts from env observation
4. **Reward functions read from kwargs** — env_reward, dq_before, dq_after come from rollout

```python
"""
DataSage — Stage 1: Cleaning GRPO Training (rollout_func edition)
==================================================================

Uses rollout_func + generate_rollout_completions to fix context mismatch:
the model sees real environment observations before generating actions,
and rewards evaluate against the same environment state.

Usage:
    python training/train_cleaning.py
"""

import json
import os
import re
import sys
import random

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.shared.config import (
    BASE_MODEL,
    HF_MODEL_REPOS,
    SPACE_URLS,
    TRAINING_CONFIGS,
    WANDB_PROJECT,
)
from training.shared.parsers import parse_cleaning_action
from environments.cleaning.client import CleaningEnv
from environments.cleaning.models import CleaningAction

ENV_URL = SPACE_URLS["cleaning"]
STAGE_CONFIG = TRAINING_CONFIGS["cleaning"]

# ── Model loading via Unsloth ────────────────────────────────────────
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── System prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a data quality agent. You clean enterprise datasets across multiple \
domains (HR, Sales, Project Management, IT Operations).

Analyze the data quality report and data preview below, then respond with a \
JSON cleaning action:
{"operation": "<op>", "column": "<col>", "value": "<val>", "params": {}}

Available operations:
- fill_null: Fill null values (value can be "median", "mode", or a specific value)
- fix_type: Fix type mismatches in a column (cast to proper type)
- remove_duplicate: Remove duplicate rows
- standardize: Standardize column values (strip whitespace, normalize case)
- trim: Trim whitespace from column values
- correct_typo: Correct typos in categorical values

Identify the most impactful issue and act."""

# ── Task descriptions (generic, env observation provides context) ────
TASK_DESCRIPTIONS = [
    "Clean the data to maximize the data quality score.",
    "Fix the most impactful data quality issue in this dataset.",
    "Analyze the DQ report and apply the best cleaning operation.",
    "This data has quality issues. Identify and fix the worst one.",
    "Improve the data quality score as much as possible with one operation.",
    "Look at the null counts, type issues, and duplicates. Fix the biggest problem.",
    "The data quality score is low. Apply the cleaning operation with the highest impact.",
    "Examine the data preview and quality report. What single operation improves quality most?",
]

# ── Dataset: simple task descriptions (env observation injected at rollout time) ──
random.seed(42)
dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": random.choice(TASK_DESCRIPTIONS)}]
        for _ in range(64)
    ]
})


# ── Prompt builder: injects env observation into prompt ──────────────
def build_prompt_with_observation(obs, task_description: str) -> str:
    """Build a complete prompt with real environment observation."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Domain: {obs.domain}\n\n"
                f"Data Quality Report:\n{obs.dq_report}\n"
                f"DQ Score: {obs.dq_score:.4f}\n\n"
                f"Columns:\n{obs.columns_info}\n\n"
                f"Data Preview:\n{obs.data_preview}\n\n"
                f"Task: {task_description}"
            )},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )


# ── Rollout function ─────────────────────────────────────────────────
def cleaning_rollout(prompts: list[str], trainer: GRPOTrainer) -> dict:
    """
    Custom rollout: reset env with seed, inject observation into prompt,
    generate completions, evaluate each against the same seeded state.
    """
    num_gens = trainer.args.num_generations
    tok = trainer.processing_class

    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    all_env_rewards = []
    all_dq_before = []
    all_dq_after = []

    for i, _prompt in enumerate(prompts):
        seed = random.randint(0, 2**31)
        task_desc = random.choice(TASK_DESCRIPTIONS)

        # 1. Reset env with seed to get observation
        with CleaningEnv(base_url=ENV_URL) as client:
            reset_result = client.reset_with_seed(seed=seed)
            obs = reset_result.observation
            dq_before = obs.dq_score

        # 2. Build prompt with real env observation
        full_prompt = build_prompt_with_observation(obs, task_desc)

        # 3. Generate N completions
        gen_prompts = [full_prompt] * num_gens
        outputs = generate_rollout_completions(trainer, gen_prompts)

        # 4. Evaluate each completion against the same seeded env state
        for out in outputs:
            text = tok.decode(out["completion_ids"], skip_special_tokens=True)

            try:
                action_dict = parse_cleaning_action(text)
                action = CleaningAction(
                    operation=action_dict.get("operation", "fill_null"),
                    column=action_dict.get("column", ""),
                    value=action_dict.get("value"),
                    params=action_dict.get("params", {}),
                )
                # Reset with SAME seed -> identical state
                with CleaningEnv(base_url=ENV_URL) as client:
                    client.reset_with_seed(seed=seed, domain=obs.domain)
                    result = client.step(action)
                    env_reward = float(result.reward or 0.0)
                    dq_after = result.observation.dq_score
            except Exception as e:
                print(f"Env error: {e}")
                env_reward = 0.0
                dq_after = dq_before

            all_prompt_ids.append(out["prompt_ids"])
            all_completion_ids.append(out["completion_ids"])
            all_logprobs.append(out["logprobs"])
            all_env_rewards.append(env_reward)
            all_dq_before.append(dq_before)
            all_dq_after.append(dq_after)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_reward": all_env_rewards,
        "dq_before": all_dq_before,
        "dq_after": all_dq_after,
    }


# ── Reward functions (read from rollout kwargs) ──────────────────────
def env_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Primary reward from environment (passed via rollout kwargs)."""
    return [float(r) for r in kwargs.get("env_reward", [0.0] * len(completions))]


def dq_improvement_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward the DQ improvement delta."""
    dq_before = kwargs.get("dq_before", [])
    dq_after = kwargs.get("dq_after", [])
    if not dq_before or not dq_after:
        return [0.0] * len(completions)
    return [max(0.0, min((after - before) * 5.0, 1.0))
            for before, after in zip(dq_before, dq_after)]


def json_format_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward well-formed JSON cleaning actions."""
    rewards = []
    for text in completions:
        if re.search(r'\{[^{}]*"operation"[^{}]*\}', text):
            try:
                match = re.search(r'\{[^{}]*"operation"[^{}]*\}', text)
                data = json.loads(match.group())
                valid_ops = {"fill_null", "fix_type", "remove_duplicate",
                             "standardize", "trim", "correct_typo"}
                if data.get("operation") in valid_ops and "column" in data:
                    rewards.append(1.0)
                elif data.get("operation") in valid_ops:
                    rewards.append(0.6)
                else:
                    rewards.append(0.3)
            except (json.JSONDecodeError, AttributeError):
                rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


# ── GRPO training config ─────────────────────────────────────────────
training_args = GRPOConfig(
    output_dir="./outputs/cleaning-grpo",
    learning_rate=STAGE_CONFIG["learning_rate"],
    num_train_epochs=STAGE_CONFIG["num_train_epochs"],
    per_device_train_batch_size=STAGE_CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=STAGE_CONFIG["gradient_accumulation_steps"],
    num_generations=STAGE_CONFIG["num_generations"],
    max_completion_length=STAGE_CONFIG["max_completion_length"],
    max_prompt_length=STAGE_CONFIG["max_prompt_length"],
    logging_steps=1,
    save_steps=50,
    bf16=True,
    use_vllm=True,
    vllm_mode="colocate",
    report_to="wandb",
    run_name="datasage-cleaning-grpo-v2",
)

os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

# ── Train ─────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[
        env_reward_fn,          # Primary: environment reward (via rollout)
        dq_improvement_reward,  # Delta: rewards actual improvement
        json_format_reward,     # Format: valid JSON output
    ],
    rollout_func=cleaning_rollout,
)

print("Starting Stage 1 (Cleaning) GRPO training v2...")
trainer.train()

# ── Save & push to Hub ───────────────────────────────────────────────
output_dir = "./outputs/cleaning-grpo-final"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Training complete! Model saved to {output_dir}")

hf_repo = HF_MODEL_REPOS["cleaning"]
print(f"Pushing to Hub: {hf_repo}")
trainer.push_to_hub(hf_repo)
print(f"Model pushed to https://huggingface.co/{hf_repo}")
```

**Step 2: Verify syntax**

Run: `cd /Users/ricalanis/Documents/dev/openenv-hackaton && python -c "import ast; ast.parse(open('training/train_cleaning.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

**Step 3: Commit**

```bash
git add training/train_cleaning.py
git commit -m "feat(training): rewrite cleaning GRPO with rollout_func for context-matched rewards"
```

---

### Task 6: Rewrite train_enrichment.py with rollout_func

**Files:**
- Modify: `training/train_enrichment.py` (major rewrite)

**Step 1: Rewrite using same pattern as Task 5**

Key differences from cleaning:
- Observation includes `available_sources`, `possible_enrichments`, `schema_info`
- Prompt builder shows available enrichment sources
- Rollout kwargs include `coverage` instead of `dq_before`/`dq_after`
- New `source_relevance_reward` checks if model picked a source mentioned in the task

```python
def build_prompt_with_observation(obs, task_description: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Domain: {obs.domain}\n\n"
                f"Schema:\n{obs.schema_info}\n\n"
                f"Available Enrichment Sources: {', '.join(obs.available_sources)}\n\n"
                f"Possible Enrichments: {', '.join(obs.possible_enrichments)}\n\n"
                f"Data Preview:\n{obs.data_preview}\n\n"
                f"Task: {task_description}"
            )},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
```

Rollout function returns: `env_reward`, `coverage`.

Reward functions:
- `env_reward_fn`: reads `env_reward` from kwargs
- `source_relevance_reward`: checks if parsed source is in `obs.available_sources` (passed via kwargs as `available_sources`)
- `json_format_reward`: same as cleaning but checks for `field_name` instead of `column`

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('training/train_enrichment.py').read())"`

**Step 3: Commit**

```bash
git add training/train_enrichment.py
git commit -m "feat(training): rewrite enrichment GRPO with rollout_func"
```

---

### Task 7: Rewrite train_answering.py with rollout_func

**Files:**
- Modify: `training/train_answering.py` (major rewrite)

**Step 1: Rewrite using same pattern**

Key differences:
- Single-step episode (no multi-turn)
- Observation includes `persona`, `persona_description`, `question`, `dataset_summary`, `column_stats`
- Prompt builder shows persona + question + data context
- Rollout kwargs include `persona_name` for persona_match_reward
- `persona_match_reward` extracts the requested persona from the prompt and checks alignment with THAT persona only (not all 3)

```python
def build_prompt_with_observation(obs, task_description: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Domain: {obs.domain}\n"
                f"Persona: {obs.persona}\n"
                f"Persona Focus: {obs.persona_description}\n\n"
                f"Question: {obs.question}\n\n"
                f"Dataset Summary:\n{obs.dataset_summary}\n\n"
                f"Column Statistics:\n{obs.column_stats}\n\n"
                f"Available Columns: {', '.join(obs.available_columns)}\n\n"
                f"Task: {task_description}"
            )},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
```

Rollout kwargs: `env_reward`, `persona_name`, `faithfulness`, `persona_relevance`.

Fixed `persona_match_reward`:

```python
def persona_match_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward alignment with the REQUESTED persona (not just any persona)."""
    persona_names = kwargs.get("persona_name", [])
    if not persona_names:
        return [0.0] * len(completions)

    from environments.shared.personas import PERSONAS, score_persona_alignment

    persona_map = {p.name: p for p in PERSONAS}
    rewards = []
    for text, p_name in zip(completions, persona_names):
        persona = persona_map.get(p_name)
        if persona:
            rewards.append(score_persona_alignment(text, persona))
        else:
            rewards.append(0.0)
    return rewards
```

Keep `patronus_reward_fn` and `json_format_reward` alongside.

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('training/train_answering.py').read())"`

**Step 3: Commit**

```bash
git add training/train_answering.py
git commit -m "feat(training): rewrite answering GRPO with rollout_func and fixed persona reward"
```

---

### Task 8: Update Existing Tests for New Signatures

**Files:**
- Modify: `tests/test_cleaning.py`
- Modify: `tests/test_enrichment.py`
- Modify: `tests/test_answering.py`

**Step 1: Update test_cleaning.py**

Ensure existing tests still call `reset()` without args (backward compatible).
Add the seeded reset tests from Task 1.
Update any assertions that reference old reward values (the reward formula changed).

**Step 2: Update test_enrichment.py**

Same pattern. Ensure `enrichment_reward` no longer expects `downstream_bucket` param.

**Step 3: Update test_answering.py**

Same pattern.

**Step 4: Run full test suite**

Run: `cd /Users/ricalanis/Documents/dev/openenv-hackaton && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/
git commit -m "test: update tests for new reward signatures and seeded reset"
```

---

### Task 9: Update Documentation

**Files:**
- Modify: `docs/changelog.md`
- Modify: `docs/plans/2026-03-07-reward-pipeline-fix-design.md` (mark implemented)

**Step 1: Add changelog entry**

Add to `docs/changelog.md`:

```markdown
## 2026-03-07 — Reward Pipeline Fix

### Changed
- Training scripts now use `rollout_func` with `generate_rollout_completions` for context-matched rewards
- All 3 environment servers support seeded reset via `/reset-with-seed` endpoint
- `reward_utils.py` simplified: removed `DOWNSTREAM_CACHE`, cleaning reward now uses improvement delta
- Persona reward checks alignment with the REQUESTED persona, not all personas

### Added
- `dq_improvement_reward`: rewards the DQ score improvement delta
- `source_relevance_reward`: rewards picking enrichment sources matching the task
- `persona_match_reward`: fixed persona alignment scoring against correct persona
- Seeded reset (`reset(seed=, domain=)`) on all 3 environment classes
- `/reset-with-seed` HTTP endpoint on all 3 environment servers
- `reset_with_seed()` method on all 3 environment clients

### Removed
- `DOWNSTREAM_CACHE` and all downstream signal mixing
- `reasoning_reward` (shallow keyword matching)
- Static `TASK_PROMPTS` in training scripts (replaced by dynamic env-observation prompts)

### Fixed
- **Context mismatch**: model now sees real environment observations before generating actions
- **Reward evaluation**: each completion is evaluated against the same seeded environment state
- **Persona reward**: now checks alignment with the requested persona, not the highest-scoring one
```

**Step 2: Mark design doc as implemented**

Change status in `docs/plans/2026-03-07-reward-pipeline-fix-design.md` from `Approved` to `Implemented`.

**Step 3: Commit**

```bash
git add docs/
git commit -m "docs: update changelog and mark reward pipeline fix as implemented"
```

---

## Execution Order and Dependencies

```
Task 1 (Cleaning seeded reset) ─┐
Task 2 (Enrichment seeded reset) ├─→ Task 4 (reward_utils) ─→ Task 5 (train_cleaning)
Task 3 (Answering seeded reset) ─┘                          ─→ Task 6 (train_enrichment)
                                                             ─→ Task 7 (train_answering)
                                                             ─→ Task 8 (update tests)
                                                             ─→ Task 9 (docs)
```

Tasks 1-3 can be done in parallel. Task 4 depends on 1-3. Tasks 5-7 depend on 4 and can be done in parallel. Tasks 8-9 depend on everything.
