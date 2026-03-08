## Phase 0 — Assumptions and current state

- **Status**: Completed
- **Summary**:
  - Identified the core environment implementation as `FSDSCleaningEnvironment` in `server/cleaning_environment.py`, built on `MCPEnvironment` from `openenv-core`.
  - Confirmed that the environment exposes an MCP tool surface (`list_tasks`, `get_task_brief`, `preview_data`, `profile_data`, `apply_cleaning_operation`, `run_quality_gates`, `submit_solution`) rather than a traditional fixed action space.
  - Located reward logic inside the environment:
    - Per-step reward from `apply_cleaning_operation` via `_quality_score` deltas plus step cost.
    - Quality-gate bonus in `run_quality_gates`.
    - Final episode reward in `submit_solution`, combining table quality, gate pass/fail, and required-operation coverage.
  - Located existing training / baseline scripts:
    - `examples/local_smoke_test.py` — low-level OpenEnv client smoke test.
    - `examples/local_agent_demo.py` — scripted baseline using a fixed cleaning policy.
    - `examples/trl_rollout_stub.py` — minimal TRL rollout + reward_fn example for OpenEnv.
    - `training_colab.py` — GRPO training notebook that connects to the HF Space and defines multiple reward functions (environment reward, JSON-format reward, methodology reward).
  - Confirmed Hugging Face / OpenEnv access patterns:
    - `server/app.py` wraps `FSDSCleaningEnvironment` into a FastAPI app via `create_app` from `openenv.core.env_server.http_server`.
    - `client.FSDSCleaningEnv` subclasses `MCPToolClient` and is used both locally and remotely (via `base_url` pointing to the HF Space, e.g. `https://israaaML-fsds-cleaning-env.hf.space`).
- **Files touched in this phase**: None (discovery only).

---

## Phase 1 — Define concrete success metrics

- **Status**: Completed (initial implementation; extensible for future refinement).
- **Summary**:
  - Created a reusable `metrics` module (`metrics.py`) that:
    - Defines `EpisodeMetrics` and `AggregateMetrics` dataclasses.
    - Provides `compute_episode_metrics(trajectory)` to summarize a single episode using:
      - total return,
      - step count,
      - inferred success flag,
      - invalid action count (based on `error` fields in tool results),
      - quality-gate pass/fail,
      - retention ratio when available.
    - Provides `aggregate_metrics(episodes)` to compute success rate, average return, average steps, and average invalid actions across many episodes.
    - Is environment-aware but trajectory-based, so it can be reused by offline evaluation scripts and future training harnesses without depending on internal server state.
  - Seeded an evaluation set module (`evaluation_tasks.py`) that:
    - Introduces an `EvaluationTask` dataclass for held-out evaluation scenarios.
    - Defines `EVAL_TASKS` with one canonical evaluation scenario per built-in task:
      - `ecommerce_mobile_baseline` → `ecommerce_mobile`
      - `subscription_churn_baseline` → `subscription_churn`
      - `delivery_eta_baseline` → `delivery_eta`
    - Documents how future agents can expand this list toward the recommended 20–50 evaluation tasks (e.g., by adding variants with different step budgets or perturbed datasets once the environment supports them).
  - Documented what “good” means in user-facing terms by adding a **What a “good” agent looks like** section to `README.md`, linking the conceptual success criteria to:
    - high task success and cumulative reward,
    - low invalid-action rate,
    - efficient step usage,
    - healthy retention and stability.
- **Files touched in this phase**:
  - `metrics.py` — new module with episode/aggregate metrics helpers.
  - `evaluation_tasks.py` — new module defining the initial evaluation scenarios.
  - `README.md` — added a metrics-focused section describing what constitutes a good agent.

---

## Phase 2 — Refine the environment for learnability

- **Status**: Completed (first pass; ready for RL agents).
- **Summary**:
  - Clarified the **observation metadata schema** for the environment:
    - Introduced `OBSERVATION_SCHEMA_VERSION = "1.0"` and a standardized metadata payload returned by `reset`, including:
      - `schema_version`, `task_id`, `task_type`, `target_column`,
      - `episode_id`, `step_count`, `max_steps`,
      - `available_tools` and a new `available_operations` list (the logical action space for `apply_cleaning_operation`).
    - This gives RL agents and analysis scripts a stable, machine-readable description of the environment state and action schema without changing existing tool semantics.
  - Made the **action space** explicit and discoverable:
    - Defined a module-level `AVAILABLE_OPERATIONS` constant that enumerates all supported cleaning operations:
      `drop_duplicates`, `replace_invalid_with_null`, `cast_numeric`, `cast_datetime`,
      `impute_numeric`, `impute_categorical`, `normalize_categories`, `clip_outliers_iqr`.
    - Exposed this list in the reset observation metadata so agents can infer the valid operation set programmatically instead of hard-coding it.
  - Improved **debuggability and episode inspection**:
    - Added a new MCP tool `render_episode(n_preview_rows=5)` that returns a human-and-agent–friendly snapshot of the current episode:
      - task metadata (task_id, title),
      - current `step_count`, `max_steps`, `total_reward`,
      - table `shape`,
      - last quality-gate report (if any),
      - full `operation_log`,
      - a small row preview from the working DataFrame.
    - Updated the `README.md` tools section to document `render_episode` and its purpose.
  - Added a lightweight **determinism check** script:
    - `examples/determinism_check.py` runs the same scripted cleaning policy twice against a local server and compares:
      - final reward,
      - pass/fail status,
      - required-operation coverage.
    - This provides a quick way to detect unintended non-determinism in episode behavior when the sequence of tool calls is fixed.
- **Files touched in this phase**:
  - `server/cleaning_environment.py` — added observation schema versioning, explicit `AVAILABLE_OPERATIONS`, richer reset metadata, and the `render_episode` tool.
  - `examples/determinism_check.py` — new script for checking deterministic behavior under a fixed policy.
  - `README.md` — documented the new `render_episode` tool in the tools list.

---

## Phase 3 — Reward system redesign

- **Status**: Completed (reward logic centralized and testable).
- **Summary**:
  - Centralized reward design into a dedicated `reward` module:
    - Added `reward.py` with:
      - `StepRewardInput` and `compute_step_reward` for dense per-step rewards based on quality-score deltas, with a small improvement margin and a floor on negative rewards.
      - `compute_quality_gate_bonus` for the gate bonus/penalty.
      - `FinalRewardInput` and `compute_final_reward` to combine table quality, gate pass/fail, and required-operation coverage at submission time.
      - `TOOL_ERROR_REWARD` constant capturing the standard penalty for tool-level errors.
  - Refactored the environment to use the new reward module:
    - `apply_cleaning_operation` now calls `compute_step_reward` instead of inlining the shaping formula, while still logging `quality_delta` for transparency.
    - `run_quality_gates` now uses `compute_quality_gate_bonus` to decide the gate reward.
    - `submit_solution` now uses `compute_final_reward` to compute the final episode reward from quality score, gate outcome, and required-op coverage.
    - `_tool_error` uses the shared `TOOL_ERROR_REWARD` constant, ensuring error penalties stay consistent if tuned later.
  - Added **hand-crafted reward checks and traces**:
    - Created `tests/test_reward.py` with small unit tests that verify:
      - positive and negative step reward behavior (including clipping),
      - quality-gate bonuses,
      - final reward composition,
      - and that tool-error rewards are negative by design.
    - Added `examples/reward_trace_demo.py`, which:
      - runs a "good" scripted policy and a deliberately "bad" minimal policy on `ecommerce_mobile`,
      - records per-step rewards, quality scores, gate rewards, and final rewards,
      - prints JSON traces so agents or notebooks can easily visualize reward trajectories for successful vs. failed episodes.
- **Files touched in this phase**:
  - `reward.py` — new centralized reward module.
  - `server/cleaning_environment.py` — updated to delegate reward logic to `reward.py`.
  - `examples/reward_trace_demo.py` — new script to inspect and compare reward traces.
  - `tests/test_reward.py` — new unit tests for reward behavior using hand-crafted scenarios.

---

## Dataset generation (learning maximization)

- **Status**: Completed
- **Summary**:
  - Created `dataset_generators.py` with synthetic generators for all three tasks:
    - `generate_mobile_ecommerce`, `generate_subscription_churn`, `generate_delivery_eta` — produce 100–2000+ row tables with configurable noise.
    - `NoiseProfile` (light/medium/heavy) controls: p_missing, p_invalid_token, p_duplicate_row, p_outlier, p_category_drift, p_string_in_numeric.
    - Realistic pathologies: missingness, invalid tokens, duplicates, outliers, category drift (whitespace/case/aliases), mixed dtypes in numeric columns.
  - **Per-episode generation for training**: `reset(seed=None)` yields a fresh random table each episode (default 500 rows, medium noise) to maximize diversity and generalization.
  - **Fixed seeds for evaluation**: `EVAL_SEEDS` and `get_eval_dataset(task_id, eval_index)` provide reproducible held-out tables. Use `reset(seed=EVAL_SEEDS[task_id][idx])` for evaluation.
  - **Debug mode**: `reset(dataset_mode="debug")` returns the original tiny static tables (12 rows) for fast debugging.
  - Wired generators into `TaskSpec.dataset_factory` via `make_dataset_factory()`. Environment `reset` passes `seed`, `dataset_mode`, `dataset_n_rows` to the factory.
  - Expanded `evaluation_tasks.py` with `eval_index` and multiple seeds per task (15 eval scenarios total).
- **Files touched**:
  - `dataset_generators.py` — new module with generators, NoiseProfile, make_dataset_factory, EVAL_SEEDS, get_eval_dataset.
  - `server/cleaning_environment.py` — replaced static dataset functions with make_dataset_factory; reset passes dataset kwargs.
  - `evaluation_tasks.py` — added eval_index, n_rows, expanded EVAL_TASKS from fixed seeds.
  - `README.md` — added "Dataset generation" section with training/eval/debug usage.

---

## Phase 4 — Baselines and evaluation harness

- **Status**: Completed
- **Summary**:
  - Implemented baseline agents in `agents.py`:
    - **RandomAgent**: Uniform random over inspect tools (profile_data, preview_data, get_task_brief), apply_cleaning_operation (random op + column), run_quality_gates, and submit_solution. Uses profile_data to discover columns for cleaning. Serves as a lower bound.
    - **HeuristicAgent**: Rule-based agent that follows the canonical cleaning policy per task (derived from required_ops). Scripted policies for ecommerce_mobile, subscription_churn, and delivery_eta.
  - Defined `Agent` protocol with `run_episode(env, task_id, max_steps, seed, **reset_kwargs)` returning a trajectory (list of step dicts with tool_name, reward, result).
  - Created `evaluate_agent.py` harness:
    - Accepts `--agent` (random | heuristic), `--base-url`, `--output`, `--episodes-per-task`, `--seed`.
    - Runs agent on EVAL_TASKS using fixed seeds from EVAL_SEEDS for reproducible evaluation.
    - Computes metrics via `compute_episode_metrics` and `aggregate_metrics`.
    - Outputs JSON (optional `-o`) and a human-readable summary (success rate, avg return, avg steps, avg invalid actions).
  - Works against local environment (http://localhost:8000) and HF-hosted environment (https://...hf.space) with minimal configuration change (just `--base-url`).
- **Files touched**:
  - `agents.py` — new module with Agent protocol, RandomAgent, HeuristicAgent, HEURISTIC_POLICIES.
  - `evaluate_agent.py` — new evaluation harness script.
  - `README.md` — added "Evaluation harness" section with usage examples.

---

## Phase 5 — Agent interface and training loop

- **Status**: Completed
- **Summary**:
  - Extended **Agent interface** in `agents.py`:
    - Added `ToolCall` TypedDict and `AgentWithAct` protocol with `act(observation, history)` → `ToolCall | None` for per-step control.
    - `run_episode` remains the primary interface; agents implement it directly or via act()-based loop.
  - Implemented **LLMAgentAdapter** for HF/LLM-based agents:
    - Wraps `generate_fn(observation, history)` → raw text and `parse_fn(text)` → `ToolCall`.
    - Default parse extracts JSON `{"tool": "...", "arguments": {...}}` from model output.
    - Implements `run_episode` by repeatedly calling `act()` until done or max_steps.
  - Created **training** package (`training/`):
    - `config.py`: `ExperimentConfig` dataclass, `from_json`, `from_yaml`, `from_file` for config loading.
    - `run_experiment.py`: Loads config, runs agent for n_episodes, collects metrics, logs to `log_dir/`, saves results to `output_dir/` as JSON.
    - Config fields: task_id, n_episodes, agent, base_url, max_steps_per_episode, log_dir, log_interval, seed, output_dir.
  - Added **configs**:
    - `configs/basic_rl.json` and `configs/basic_rl.yaml` with default experiment settings.
  - **Command**: `python -m fsds_cleaning_env.training.run_experiment --config configs/basic_rl.json`
- **Files touched**:
  - `agents.py` — ToolCall, AgentWithAct, LLMAgentAdapter.
  - `training/__init__.py`, `training/config.py`, `training/run_experiment.py` — new training package.
  - `configs/basic_rl.json`, `configs/basic_rl.yaml` — example configs.
  - `pyproject.toml` — added fsds_cleaning_env.training package.
  - `README.md` — "Training harness" section.

---

## Phase 6 — Curriculum and task diversity

- **Status**: Completed
- **Summary**:
  - Created `curriculum.py` with a `CurriculumScheduler` that drives progressive difficulty training:
    - Three `DifficultyLevel` stages (`easy`, `medium`, `hard`) each with distinct `NoiseProfile`, row count, step budget, and promotion threshold.
    - `easy`:  `NoiseProfile.light()`, 100 rows, 22 steps, promote at 70% success over 10 episodes.
    - `medium`: `NoiseProfile.medium()`, 500 rows, 18 steps, promote at 65% success over 15 episodes.
    - `hard`:  `NoiseProfile.heavy()`, 1000 rows, 15 steps (terminal level).
    - Two task-sampling modes: `round_robin` (cycle tasks in order, default) and `random` (uniform pick).
    - `CurriculumTask` dataclass carries all reset parameters for one episode; its `reset_kwargs()` method can be unpacked directly into `env.reset()`.
    - `CurriculumScheduler.summary()` returns a serializable dict with current level, window success rate, and the full promotion history.
  - Updated `training/config.py`:
    - Added four new curriculum fields: `curriculum: bool`, `curriculum_task_ids`, `curriculum_mode`, `curriculum_start_level`.
    - Refactored `from_json` / `from_yaml` into a shared `_parse` helper to avoid code duplication.
  - Updated `training/run_experiment.py`:
    - When `config.curriculum` is `True`, constructs a `CurriculumScheduler` and uses it to select `task_id`, `max_steps`, `n_rows`, and `noise_profile_override` for each episode.
    - Per-episode log entries now include `task_id`, `difficulty`, and `promoted_to` fields.
    - Progress line includes current difficulty level name.
    - Final results JSON includes a `curriculum` key with the scheduler summary.
  - Added `configs/curriculum_rl.json` and `configs/curriculum_rl.yaml`:
    - 120 episodes, heuristic agent, round-robin over all three tasks, starting at `easy`.
  - Added `examples/curriculum_demo.py`:
    - Offline simulation mode (no server needed) demonstrating difficulty promotions over 60 episodes.
    - Live mode (`--live`) that runs real environment episodes against a local or remote server.
    - Confirmed promotions: easy → medium at ep=10 (100% rate), medium → hard at ep=35 (66.7% rate).
- **Files touched in this phase**:
  - `curriculum.py` — new module with `DifficultyLevel`, `DIFFICULTY_LEVELS`, `CurriculumTask`, `CurriculumScheduler`.
  - `training/config.py` — added curriculum config fields; refactored shared `_parse` helper.
  - `training/run_experiment.py` — curriculum-aware training loop.
  - `configs/curriculum_rl.json`, `configs/curriculum_rl.yaml` — new curriculum experiment configs.
  - `examples/curriculum_demo.py` — offline + live demo script.
  - `__init__.py` — exported `CurriculumScheduler`, `CurriculumTask`, `DifficultyLevel`.

---

## Phase 7 — SFT-first, RL-second (if demonstrations exist)

- **Status**: Completed
- **Summary**:
  - Created `demonstrations.py` — the complete demonstration pipeline for SFT data generation:
    - `DemoStep` and `Demonstration` dataclasses capture full episode trajectories with per-step rewards and success flags.
    - `DemonstrationCollector` wraps any agent (defaults to `HeuristicAgent`) and collects `n_per_task` episodes per task, supporting optional `noise_profile_override` and `dataset_n_rows` for curriculum-aligned collection.
    - `save_demonstrations` / `load_demonstrations` for JSON persistence so demonstrations can be cached and reused across training runs.
    - `demo_to_step_examples(demo)` — converts one episode into N step-level `(prompt, completion)` pairs where the prompt contains the system prompt + cumulative history and the completion is the JSON action; matches the GRPO format in `training_colab.py`.
    - `demo_to_episode_example(demo)` — converts one episode into a single multi-turn conversation suitable for full-episode SFT.
    - `build_sft_dataset(demos, mode="step"|"episode")` — builds a Hugging Face `Dataset` from demonstrations, with optional `successful_only` filtering so the model only learns from winning trajectories.
    - `build_sft_dataset_from_heuristic(...)` — one-liner convenience function for fast dataset creation.
    - `demo_stats(demos)` — serializable summary statistics (success rate, avg steps, avg reward, breakdown by task).
  - Created `training_sft.py` — Colab-style SFT training script (mirrors structure of `training_colab.py`):
    - Cells: install → configure → collect/load demos → build SFT dataset → load Unsloth model → format with `apply_chat_template` → train with `trl.SFTTrainer` → save checkpoint.
    - Supports both `"step"` and `"episode"` SFT modes via `SFT_MODE` config variable.
    - `COLLECT_FRESH` flag: `True` = collect fresh demos from HF Space; `False` = load from `DEMO_PATH`.
    - Saves SFT adapter to `SFT_FINAL_DIR` and prints explicit instructions to plug the checkpoint path into `training_colab.py` for RL warm-start.
  - Updated `training_colab.py` — added SFT warm-start annotation:
    - Added a prominent comment block at the `MODEL_NAME` line explaining the SFT-first strategy and showing the exact one-line change needed to warm-start GRPO from an SFT checkpoint.
    - Commented-out line: `# MODEL_NAME = "./data-cleaning-sft-final"` for easy activation.
  - Added `configs/sft_config.json` documenting all SFT hyperparameters and paths in one place.
  - Verified with offline unit test: step-level conversion, episode-level conversion, JSON round-trip serialization, and `demo_stats` all pass.
- **Files touched in this phase**:
  - `demonstrations.py` — new module with DemoStep, Demonstration, DemonstrationCollector, SFT formatters, dataset builders, serialization helpers.
  - `training_sft.py` — new Colab-style SFT training script.
  - `training_colab.py` — added SFT warm-start comment and commented-out MODEL_NAME line.
  - `configs/sft_config.json` — new SFT configuration file.

---

## Phase 8 — Documentation and agent onboarding

- **Status**: Completed
- **Summary**:
  - Created `AGENT_GUIDE.md` — a comprehensive, self-contained onboarding reference (13 sections, ~300 lines):
    - **What the environment tests**: task table with task_id, domain, and target column.
    - **Connecting**: local server setup, HF Space URL, full reset kwarg table (task_id, seed, dataset_mode, dataset_n_rows, noise_profile_override).
    - **Episode lifecycle**: visual flow diagram (reset → inspect → clean → gates → submit) with step budget notes.
    - **Tool reference**: complete input/output JSON schemas for all 8 tools (get_task_brief, profile_data, preview_data, apply_cleaning_operation, run_quality_gates, submit_solution, render_episode, list_tasks).
    - **Action space table**: all 8 operations with column-required flag and one-line description.
    - **Reward structure**: per-step formula, gate bonus/penalty, final reward formula, error penalty — all with exact numbers.
    - **Observation schema**: annotated JSON of the reset observation dict.
    - **Writing an LLM agent**: minimal `LLMAgentAdapter` usage pattern, expected JSON output format, system prompt location, suggested episode strategy.
    - **Evaluation protocol**: harness commands, how EVAL_SEEDS work, metric definitions, target baselines table (Random vs Heuristic vs Good LLM).
    - **Curriculum training**: scheduler API, difficulty table, config command.
    - **SFT → RL pipeline**: three-step pipeline diagram with rationale.
    - **Common mistakes table**: 8 failure modes with symptom and fix.
    - **File map**: annotated tree of every module in the project.
  - Created `examples/minimal_agent.py` — the simplest complete working agent (~130 lines):
    - Every step commented to explain the "why" (reset, profile, apply required_ops, run gates, submit).
    - CLI: `--base-url` and `--task` flags for easy local/remote testing.
    - Prints quality score, gate results, final reward, and success flag.
    - Gracefully handles early episode termination (step budget exhausted).
  - Updated `README.md`:
    - Added link to `AGENT_GUIDE.md` at the top with a "New here?" call-out.
    - Added a **Quick start** section (4 commands: install + server, minimal agent, eval harness, curriculum experiment).
- **Files touched in this phase**:
  - `AGENT_GUIDE.md` — new comprehensive agent onboarding reference.
  - `examples/minimal_agent.py` — new minimal runnable agent example.
  - `README.md` — added guide link and Quick start section.

