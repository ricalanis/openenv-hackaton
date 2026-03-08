# FSDS Cleaning Environment — Agent Guide

A complete reference for writing agents that solve the data-cleaning tasks.
Start with **Quick start** if you just want something running, then read the sections that apply to your use-case.

---

## Table of contents

1. [What the environment tests](#1-what-the-environment-tests)
2. [Connecting to the environment](#2-connecting-to-the-environment)
3. [Episode lifecycle](#3-episode-lifecycle)
4. [Tool reference](#4-tool-reference)
5. [Action space — cleaning operations](#5-action-space--cleaning-operations)
6. [Reward structure](#6-reward-structure)
7. [Observation / reset schema](#7-observation--reset-schema)
8. [Writing an LLM agent](#8-writing-an-llm-agent)
9. [Evaluation protocol](#9-evaluation-protocol)
10. [Curriculum training](#10-curriculum-training)
11. [SFT → RL pipeline](#11-sft--rl-pipeline)
12. [Common mistakes](#12-common-mistakes)
13. [File map](#13-file-map)

---

## 1. What the environment tests

The agent receives a **messy business table** and must clean it to Silver quality before a downstream ML model trains on it.  Three task domains are included:

| `task_id`            | Table                           | Target column             |
|----------------------|---------------------------------|---------------------------|
| `ecommerce_mobile`   | Mobile-session conversion data  | `converted` (binary)      |
| `subscription_churn` | Subscriber attributes           | `churned` (binary)        |
| `delivery_eta`       | Last-mile delivery routes       | `delivery_time_minutes`   |

Each episode the agent must:

1. **Inspect** the dataset (optional but highly recommended).
2. **Clean** it by applying operations from the action space.
3. **Validate** by running quality gates.
4. **Submit** the cleaned table.

Success means all quality gates pass and the final reward > 0.5.

---

## 2. Connecting to the environment

### Local server (recommended for development)

```bash
# Install and start the server
pip install -e .
uvicorn fsds_cleaning_env.server.app:app --host 0.0.0.0 --port 8000
```

```python
from fsds_cleaning_env.client import FSDSCleaningEnv

with FSDSCleaningEnv(base_url="http://localhost:8000").sync() as env:
    env.reset(task_id="ecommerce_mobile")
    print(env.call_tool("get_task_brief"))
```

### Hugging Face Space

```python
with FSDSCleaningEnv(base_url="https://israaaML-fsds-cleaning-env.hf.space").sync() as env:
    env.reset(task_id="ecommerce_mobile")
```

### Reset options

| Kwarg                    | Type              | Default      | Effect                                      |
|--------------------------|-------------------|--------------|---------------------------------------------|
| `task_id`                | `str`             | required     | Which task to load                          |
| `seed`                   | `int \| None`     | `None`       | Fixed seed → reproducible table; `None` → fresh random table each episode |
| `dataset_mode`           | `"debug" \| None` | `None`       | `"debug"` uses original 12-row static table |
| `dataset_n_rows`         | `int`             | `500`        | Row count for synthetic generation          |
| `noise_profile_override` | `NoiseProfile`    | medium       | Override noise level (`NoiseProfile.light()` / `.medium()` / `.heavy()`) |

---

## 3. Episode lifecycle

```
reset(task_id, seed=None)
    │
    ▼
[optional] get_task_brief()   ← understand the task objective
[optional] profile_data()     ← learn column types, missing counts, etc.
[optional] preview_data()     ← see sample rows
    │
    ▼  (repeat up to max_steps)
apply_cleaning_operation(operation, column=None)
    │
    ├─ returns: {quality_score, quality_delta, reward, operation_log, …}
    │
    ▼
run_quality_gates()           ← check pass/fail; triggers gate bonus/penalty
    │
    ▼
submit_solution()             ← ends episode; returns final_reward, done=True
```

**Step budget**: 18 steps by default (22 in `easy` curriculum stage, 15 in `hard`).
**Tip**: one `run_quality_gates` + one `submit_solution` = 2 steps, so plan to finish cleaning in `max_steps - 2`.

---

## 4. Tool reference

### `get_task_brief()`

Returns the task description, objective, target column, and required operations.

```json
{
  "task_id": "ecommerce_mobile",
  "title": "Mobile conversion cleaning",
  "objective": "Prepare a mobile conversion table for downstream modeling.",
  "target_column": "converted",
  "task_type": "classification",
  "required_ops": [
    {"operation": "drop_duplicates"},
    {"operation": "cast_numeric", "column": "items_in_cart"},
    ...
  ],
  "notes": ["...", "..."]
}
```

### `profile_data()`

Column-level statistics for the current working table.

```json
{
  "shape": [512, 8],
  "columns": ["session_id", "device_os", ...],
  "dtypes": {"session_id": "int64", "device_os": "object", ...},
  "missing_counts": {"device_os": 12, "items_in_cart": 0, ...},
  "missing_pct": {"device_os": 2.34, ...},
  "n_duplicates": 15,
  "invalid_token_counts": {"country": 8, ...}
}
```

### `preview_data(n=5)`

Returns `n` sample rows as a list of row dicts.

### `apply_cleaning_operation(operation, column=None, **kwargs)`

Applies one operation to the working table.

```json
{
  "status": "ok",
  "operation": "cast_numeric",
  "column": "items_in_cart",
  "quality_score": 0.74,
  "quality_delta": 0.06,
  "reward": 0.04,
  "rows_affected": 23,
  "operation_log": [...]
}
```

Returns an `error` field (and a negative reward) for invalid calls.

### `run_quality_gates()`

```json
{
  "passed": true,
  "reward": 0.15,
  "gate_results": {
    "test_no_missing": {"passed": true, "details": "0 nulls"},
    "test_no_duplicates": {"passed": true},
    "test_target_preserved": {"passed": true},
    "test_retention": {"passed": true, "retention_ratio": 0.97},
    "test_dtype_alignment": {"passed": true},
    "test_stability": {"passed": true, "cv": 0.012}
  }
}
```

### `submit_solution()`

Ends the episode.

```json
{
  "done": true,
  "final_reward": 0.82,
  "quality_score": 0.91,
  "gate_passed": true,
  "required_op_coverage": 0.90,
  "operation_log": [...]
}
```

### `render_episode(n_preview_rows=5)`

Human-friendly snapshot: step count, total reward so far, last gate report, operation log, and a small data preview.  Useful for debugging; does **not** count as a cleaning step.

### `list_tasks()`

Lists all available task IDs with short descriptions.

---

## 5. Action space — cleaning operations

All operations are called via `apply_cleaning_operation(operation=..., column=...)`.

| Operation                 | `column` required | Description                                              |
|---------------------------|:-----------------:|----------------------------------------------------------|
| `drop_duplicates`         | no                | Remove exact duplicate rows                              |
| `replace_invalid_with_null` | yes             | Replace invalid tokens (`""`, `"unknown"`, `"N/A"`, etc.) with `NaN` |
| `cast_numeric`            | yes               | Parse string column to float (coerces unparseable → NaN) |
| `cast_datetime`           | yes               | Parse string column to datetime                          |
| `impute_numeric`          | yes               | Fill numeric NaN with median (default) or mean (`strategy="mean"`) |
| `impute_categorical`      | yes               | Fill categorical NaN with mode                           |
| `normalize_categories`    | yes               | Strip whitespace and lowercase all values                |
| `clip_outliers_iqr`       | yes               | Clip values outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`    |

**Invalid token set**: `{"", " ", "unknown", "UNKNOWN", "n/a", "N/A", "null", "NULL", "?", "--"}`

---

## 6. Reward structure

### Per-step reward (from `apply_cleaning_operation`)

```
reward = max(-0.15, (quality_after − quality_before) − 0.02)
```

A quality improvement > 0.02 gives a positive reward. Neutral or harmful operations give a small negative reward (floor: −0.15).

### Gate reward (from `run_quality_gates`)

```
reward = +0.15 if all gates pass, −0.10 otherwise
```

### Final reward (from `submit_solution`)

```
final_reward = 0.45 × quality_score
             + 0.30 × gate_passed   (1.0 or 0.0)
             + 0.25 × required_op_coverage
```

`required_op_coverage` = fraction of the task's required operations that were applied.

### Error reward

Any tool call that returns an `error` field gives a fixed penalty of **−0.20**.

### Maximizing reward

The ideal trajectory:
1. **Inspect** (0 reward cost — tools like `profile_data` have no step reward).
2. **Apply all required operations** in a sensible order (positive deltas).
3. **Run quality gates** after cleaning (gate bonus).
4. **Submit** once gates pass.

---

## 7. Observation / reset schema

`env.reset()` returns an observation dict:

```json
{
  "schema_version": "1.0",
  "task_id": "ecommerce_mobile",
  "task_type": "classification",
  "target_column": "converted",
  "episode_id": "uuid-...",
  "step_count": 0,
  "max_steps": 18,
  "available_tools": ["get_task_brief", "profile_data", ...],
  "available_operations": ["drop_duplicates", "replace_invalid_with_null", ...]
}
```

After each tool call, the result dict is the next observation.  Track `step_count` vs `max_steps` to know when the budget runs out.

---

## 8. Writing an LLM agent

### Minimal pattern

```python
from fsds_cleaning_env.agents import LLMAgentAdapter

def my_generate_fn(observation, history):
    # Build a prompt from observation + history, call your model.
    # Return the model's raw text output.
    return model.generate(build_prompt(observation, history))

agent = LLMAgentAdapter(generate_fn=my_generate_fn)

with FSDSCleaningEnv(base_url=...).sync() as env:
    trajectory = agent.run_episode(env, task_id="ecommerce_mobile", max_steps=18)
```

### Expected output format

The model must emit exactly one JSON object per turn:

```json
{"tool": "<tool_name>", "arguments": {"operation": "<op>", "column": "<col>"}}
```

Examples:

```json
{"tool": "profile_data", "arguments": {}}
{"tool": "apply_cleaning_operation", "arguments": {"operation": "drop_duplicates"}}
{"tool": "apply_cleaning_operation", "arguments": {"operation": "cast_numeric", "column": "items_in_cart"}}
{"tool": "apply_cleaning_operation", "arguments": {"operation": "impute_numeric", "column": "items_in_cart", "strategy": "median"}}
{"tool": "run_quality_gates", "arguments": {}}
{"tool": "submit_solution", "arguments": {}}
```

### System prompt

The system prompt in `demonstrations.py` (`SYSTEM_PROMPT`) is the canonical prompt used for both SFT and GRPO training.  Copy it verbatim to ensure consistency with the fine-tuned checkpoints.

### Suggested episode strategy (for prompt engineering)

```
Turn 1:  profile_data               ← always start here
Turn 2:  get_task_brief             ← read required operations
Turn 3+: apply operations in order  ← required_ops first, then optional
Turn N-1: run_quality_gates         ← check before submitting
Turn N:  submit_solution
```

---

## 9. Evaluation protocol

Use the built-in evaluation harness with **held-out fixed seeds** so results are reproducible:

```bash
# Heuristic baseline (upper bound for scripted policy)
python -m fsds_cleaning_env.evaluate_agent --agent heuristic \
    --base-url http://localhost:8000 -o results/heuristic.json

# Random baseline (lower bound)
python -m fsds_cleaning_env.evaluate_agent --agent random \
    --base-url http://localhost:8000 -o results/random.json
```

Evaluation tasks are defined in `evaluation_tasks.py` — 5 fixed seeds × 3 tasks = 15 scenarios.  Each seed produces a deterministic table via `EVAL_SEEDS`.

**Metrics** (computed by `metrics.py`):

| Metric              | Definition                                                     |
|---------------------|----------------------------------------------------------------|
| `success_rate`      | Fraction of episodes where quality gates passed                |
| `avg_return`        | Mean cumulative reward per episode                             |
| `avg_steps`         | Mean number of tool calls per episode                          |
| `avg_invalid_actions` | Mean number of error-returning tool calls per episode        |

**Target baselines**:

| Agent      | Success rate | Avg return |
|------------|-------------|------------|
| Random     | ~5–15%       | ~0.1–0.2   |
| Heuristic  | ~95–100%     | ~0.7–0.9   |
| Good LLM   | ≥80%         | ≥0.6       |

---

## 10. Curriculum training

The `CurriculumScheduler` gradually increases difficulty so agents see stable rewards early in training:

```
easy   → light noise, 100 rows, 22 steps   → promote at ≥70% success / 10 episodes
medium → medium noise, 500 rows, 18 steps  → promote at ≥65% success / 15 episodes
hard   → heavy noise, 1000 rows, 15 steps  → terminal level
```

```python
from fsds_cleaning_env.curriculum import CurriculumScheduler

scheduler = CurriculumScheduler(start_level="easy")

for ep in range(n_episodes):
    cur = scheduler.next_task(seed=ep)
    trajectory = agent.run_episode(env, **cur.reset_kwargs(), max_steps=cur.max_steps)
    scheduler.record_episode(success=episode_succeeded(trajectory))
```

Config-driven: `python -m fsds_cleaning_env.training.run_experiment --config configs/curriculum_rl.json`

---

## 11. SFT → RL pipeline

The recommended training pipeline:

```
1. Collect expert demonstrations (HeuristicAgent)
        ↓
2. SFT on step-level (prompt, action) pairs   ← training_sft.py
        ↓
3. GRPO / RL fine-tuning from SFT checkpoint  ← training_colab.py  (set MODEL_NAME)
```

**Why SFT first?** The base model doesn't know the JSON format or the correct operation ordering.  SFT teaches these quickly from ~60 perfect demonstrations.  GRPO then optimizes for reward rather than imitation, pushing performance beyond what the heuristic can achieve.

---

## 12. Common mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Applying `cast_numeric` before `replace_invalid_with_null` | Many NaN after cast | Replace invalid tokens first, then cast |
| Skipping `run_quality_gates` before `submit_solution` | Miss the gate bonus (+0.15) | Always run gates as the second-to-last step |
| Applying operations on the **target column** | Gate fails (`test_target_preserved`) | Never touch `converted`, `churned`, or `delivery_time_minutes` |
| Using `impute_numeric` on a still-string column | Tool error, −0.20 penalty | Cast to numeric first |
| Calling `clip_outliers_iqr` before imputation | Outlier detection distorted by NaN | Impute missing values before clipping outliers |
| Too many inspect calls | Steps wasted, budget runs out | One `profile_data` is usually enough |
| Submitting before gates pass | Low final reward | Run `run_quality_gates`; fix remaining failures first |
| Using `dataset_mode="debug"` for training | Agent overfit to 12 static rows | Use `seed=None` (default) for training; fixed seeds for eval only |

---

## 13. File map

```
fsds_cleaning_env/
├── server/
│   ├── cleaning_environment.py   Core environment (MCP tools, reward logic, TaskSpecs)
│   └── app.py                    FastAPI wrapper for OpenEnv / HF Space deployment
│
├── client.py                     FSDSCleaningEnv client (sync + async)
├── agents.py                     RandomAgent, HeuristicAgent, LLMAgentAdapter
├── metrics.py                    EpisodeMetrics, AggregateMetrics, compute_*
├── reward.py                     Centralized reward formulas (step, gate, final)
├── dataset_generators.py         Synthetic data generators, NoiseProfile, EVAL_SEEDS
├── evaluation_tasks.py           EVAL_TASKS — 15 held-out evaluation scenarios
├── curriculum.py                 CurriculumScheduler — progressive difficulty
├── demonstrations.py             Demo collection, SFT formatting, dataset builders
│
├── training/
│   ├── config.py                 ExperimentConfig (JSON/YAML loader)
│   └── run_experiment.py         Config-driven training loop (curriculum-aware)
│
├── training_colab.py             GRPO / RL training script (Colab)
├── training_sft.py               SFT training script (Colab)
│
├── configs/
│   ├── basic_rl.json / .yaml     Standard single-task experiment
│   ├── curriculum_rl.json / .yaml Curriculum experiment (all 3 tasks, easy→hard)
│   └── sft_config.json           SFT hyperparameters and paths
│
├── examples/
│   ├── minimal_agent.py          Simplest working agent — start here
│   ├── local_agent_demo.py       Scripted baseline demo
│   ├── curriculum_demo.py        Curriculum scheduler demo (offline + live)
│   ├── reward_trace_demo.py      Compare reward trajectories of good vs bad policy
│   ├── determinism_check.py      Verify episode determinism under fixed policy
│   └── trl_rollout_stub.py       Minimal TRL rollout pattern
│
├── tests/
│   └── test_reward.py            Unit tests for reward module
│
├── AGENT_GUIDE.md                ← you are here
└── README.md                     HF Space landing page
```
