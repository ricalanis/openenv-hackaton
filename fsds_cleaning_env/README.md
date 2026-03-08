---
title: FSDS Cleaning Environment
emoji: 🧼
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl
  - data-science
  - cleaning
  - fsds
---

# FSDS Cleaning Environment

An HF-ready OpenEnv environment for the **cleaning / Silver-layer** of Full-Stack Data Science.

> **New here?** Read **[AGENT_GUIDE.md](AGENT_GUIDE.md)** — tool reference, reward formulas, episode walkthrough, common mistakes, and the full training pipeline.

## What this environment tests

The agent must turn a messy business table into a trustworthy Silver table by:

1. profiling the dataset,
2. identifying duplicates / invalid tokens / schema issues,
3. applying cleaning operations,
4. passing quality gates,
5. submitting the cleaned result.

The environment is inspired by:
- **FSDS**: Bronze → Silver → Gold, with a DS Agent plus QA gates.
- **VDSAgents**: Explore-Agent + PCS-style unit tests and perturbation-minded validation.
- **Data Interpreter**: progressive, tool-driven, multi-step execution instead of one-shot code.

## Tasks included

- `ecommerce_mobile` — clean a mobile conversion table.
- `subscription_churn` — clean a subscriber churn table.
- `delivery_eta` — clean a last-mile delivery table for ETA modeling.

## Quick start

```bash
# 1. Start the server
pip install -e .
uvicorn fsds_cleaning_env.server.app:app --port 8000

# 2. Run the minimal agent example (in another shell)
python examples/minimal_agent.py

# 3. Run the evaluation harness
python -m fsds_cleaning_env.evaluate_agent --agent heuristic

# 4. Run a curriculum training experiment
python -m fsds_cleaning_env.training.run_experiment --config configs/curriculum_rl.json
```

See [AGENT_GUIDE.md](AGENT_GUIDE.md) for the full reference.

## Dataset generation (maximize RL learning)

Tables are generated **per episode** by default (500 rows, medium noise) so the agent sees diverse data each run. This improves generalization.

- **Training**: `env.reset(task_id="ecommerce_mobile", seed=None)` — fresh random table each episode.
- **Evaluation**: `env.reset(task_id="ecommerce_mobile", seed=42)` — fixed seed for reproducible held-out data.
- **Debug**: `env.reset(task_id="ecommerce_mobile", dataset_mode="debug")` — original tiny static table (12 rows).

Optional kwargs: `dataset_n_rows` (override size), `dataset_mode="debug"` (use static data). See `dataset_generators.py` for `NoiseProfile`, `SIZE_*`, and `get_eval_dataset()`.

## Tools

- `list_tasks()`
- `get_task_brief()`
- `preview_data(n=5)`
- `profile_data()`
- `get_operation_history()`
- `apply_cleaning_operation(operation, column=None, strategy="median")`
- `run_quality_gates()`
- `submit_solution()`
- `render_episode(n_preview_rows=5)` — human-friendly snapshot with step count, total reward, and a small preview.

## Reward design

Each cleaning action receives a small dense reward based on **quality score improvement**:

- positive reward when the table gets cleaner,
- small step cost to discourage unnecessary actions,
- gate bonus for passing quality checks,
- final reward combining table quality, gate pass/fail, and coverage of required cleaning operations.

This satisfies the hackathon preference for coherent rewards and observable improvement.

## Quality gates

The built-in QA / PCS layer checks:

- no unresolved missing values outside the target,
- no duplicate rows,
- target column preserved,
- row retention above threshold,
- dtype alignment,
- simple stability probe via repeated downstream model scoring.

## What a “good” agent looks like

At evaluation time, we care less about sounding smart and more about concrete, testable behavior. In practice, a good agent for this environment has:

- **High task success**: consistently passes the quality gate on held-out episodes.
- **Strong returns**: high cumulative reward per episode (from dense step rewards + gate + final reward).
- **Low invalid action rate**: few or no tool calls that trigger `error` responses.
- **Good efficiency**: solves tasks in relatively few steps without unnecessary operations.
- **Healthy retention and stability**: preserves enough rows, aligns dtypes, and passes the stability probe.

These metrics can be computed from trajectories using the `metrics` module and evaluated on the scenarios in `evaluation_tasks.py`.

## Local development

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

In another shell:

```bash
python examples/local_agent_demo.py
```

## Deploy to Hugging Face

```bash
openenv push
```

## Use from a client

```python
from fsds_cleaning_env import FSDSCleaningEnv

with FSDSCleaningEnv(base_url="https://YOUR-SPACE.hf.space").sync() as env:
    env.reset(task_id="ecommerce_mobile")
    print(env.call_tool("get_task_brief"))
    print(env.call_tool("profile_data"))
    print(env.call_tool("apply_cleaning_operation", operation="drop_duplicates"))
    print(env.call_tool("run_quality_gates"))
    print(env.call_tool("submit_solution"))
```

## Evaluation harness

Run baseline agents on the held-out evaluation set:

```bash
# Local environment (start server first: uvicorn server.app:app --port 8000)
python -m fsds_cleaning_env.evaluate_agent --agent heuristic --base-url http://localhost:8000
python -m fsds_cleaning_env.evaluate_agent --agent random -o results.json

# HF-hosted environment
python -m fsds_cleaning_env.evaluate_agent --agent heuristic --base-url https://YOUR-SPACE.hf.space
```

Agents: `RandomAgent` (uniform random over tools), `HeuristicAgent` (rule-based canonical policy). Output: success rate, avg return, avg steps, invalid actions; optional JSON file.

## Training harness

Run config-driven training experiments:

```bash
# Start server first: uvicorn fsds_cleaning_env.server.app:app --port 8000
python -m fsds_cleaning_env.training.run_experiment --config configs/basic_rl.json
# Or with YAML (requires: pip install pyyaml)
python -m fsds_cleaning_env.training.run_experiment --config configs/basic_rl.yaml
```

Config: `task_id`, `n_episodes`, `agent` (random | heuristic), `base_url`, `max_steps_per_episode`, `log_dir`, `log_interval`, `seed`, `output_dir`. Results are logged to `log_dir/` and saved as JSON in `output_dir/`. For full GRPO/TRL training with LLMs, see `training_colab.py`.

## Curriculum training

The `CurriculumScheduler` in `curriculum.py` drives progressive difficulty across three stages:

| Stage  | Noise   | Rows | Steps | Promote at              |
|--------|---------|------|-------|-------------------------|
| easy   | light   | 100  | 22    | ≥70% success / 10 eps   |
| medium | medium  | 500  | 18    | ≥65% success / 15 eps   |
| hard   | heavy   | 1000 | 15    | terminal level          |

**Quick start** — run the curriculum experiment config:

```bash
# Start server first
python -m fsds_cleaning_env.training.run_experiment --config configs/curriculum_rl.json
```

**Demo (no server required):**

```bash
python examples/curriculum_demo.py --n-episodes 60
# Live run (requires server):
python examples/curriculum_demo.py --live --base-url http://localhost:8000 --n-episodes 30
```

**Programmatic usage:**

```python
from fsds_cleaning_env.curriculum import CurriculumScheduler

scheduler = CurriculumScheduler(
    task_ids=["ecommerce_mobile", "subscription_churn", "delivery_eta"],
    mode="round_robin",   # or "random"
    start_level="easy",
)

for ep in range(n_episodes):
    cur = scheduler.next_task(seed=ep)
    trajectory = agent.run_episode(env, **cur.reset_kwargs(), max_steps=cur.max_steps)
    promoted = scheduler.record_episode(success=trajectory_success(trajectory))
    if promoted:
        print(f"Promoted to {scheduler.level_name}!")
```

`scheduler.summary()` returns a JSON-serialisable dict with the current level, rolling success rate, and full promotion history — automatically saved in run results when using the training harness.

## SFT-first, RL-second (Phase 7)

Expert trajectories from the `HeuristicAgent` are used to SFT-warm-start the model before GRPO reinforcement learning.  A warm-start model already knows the JSON action format and the correct *inspect → clean → validate* methodology, so GRPO converges much faster.

**Step 1 — Collect demonstrations and train SFT model (Colab)**

Open `training_sft.py` in Colab and run all cells.  It will:
1. Connect to the HF Space and collect expert trajectories.
2. Build a Hugging Face Dataset of step-level `(prompt, completion)` pairs.
3. Fine-tune via `trl.SFTTrainer` and save the adapter to `./data-cleaning-sft-final/`.

**Step 2 — Use the SFT checkpoint as the RL warm-start**

In `training_colab.py`, change one line:

```python
# Before (trains from scratch):
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"

# After (warm-start from SFT):
MODEL_NAME = "./data-cleaning-sft-final"
```

**Programmatic demo collection (no Colab)**

```python
from fsds_cleaning_env.demonstrations import DemonstrationCollector, build_sft_dataset, save_demonstrations
from fsds_cleaning_env import FSDSCleaningEnv

with FSDSCleaningEnv(base_url="http://localhost:8000").sync() as env:
    demos = DemonstrationCollector(env).collect(
        task_ids=["ecommerce_mobile", "subscription_churn", "delivery_eta"],
        n_per_task=20,
    )

save_demonstrations(demos, "demos/expert_demos.json")
dataset = build_sft_dataset(demos, mode="step", successful_only=True)
print(f"{len(dataset)} SFT training examples")
```

Config reference: `configs/sft_config.json`.

## Minimal TRL integration

See `examples/trl_rollout_stub.py` for a compact `rollout_func` pattern that forwards `env_reward` into a TRL reward function.
