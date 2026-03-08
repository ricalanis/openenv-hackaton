## Objective

This project aims to **maximize the learning signal (RL)** that a **Hugging Face–hosted environment** can provide to any agent (human-designed policies, HF language models, or RL policies).

The core idea:

- **Environment** = what the agent experiences.
- **Reward** = what the agent is encouraged to repeat.
- **Training loop** = how strongly it adapts.
- **Evaluation harness** = how we know it is actually improving.

If environment, reward, or evaluation are weak, more training and bigger models will not help much. This plan is organized to improve those pieces first.

---

## High-level priorities

In order of impact:

1. **Environment clarity and stability** (observations, actions, terminals, determinism).
2. **Reward correctness and robustness** (aligned, dense-enough, hard to game).
3. **Evaluation and baselines** (can we measure improvement and beat trivial policies?).
4. **Training harness and agent interface** (easy to plug in different agents/models).
5. **Curriculum, SFT, and scaling** (only once 1–4 are solid).

Agents (humans and code assistants) should work through the phases below roughly in this order.

---

## Phase 0 — Assumptions and current state

- **Goal**: Make sure all collaborators and agents share the same mental model of the task.

- **Actions for agents:**
  - [ ] Locate the environment implementation (Python files that implement `reset`, `step`, and define observations/actions).
  - [ ] Locate any existing reward logic (where numerical rewards are computed).
  - [ ] Locate any training scripts or notebooks that already run episodes or RL.
  - [ ] Confirm how the Hugging Face environment is accessed (local class vs remote API vs `gym`-style wrapper).

Document these findings at the top of this file or in a short `ENV_NOTES.md` so future agents can jump in quickly.

---

## Phase 1 — Define concrete success metrics

- **Goal**: Turn “good agent” into measurable quantities.

- **Recommended metrics (pick the ones that fit):**
  - **Task success**: success rate on held-out episodes.
  - **Return**: average episode return.
  - **Validity**: number or rate of invalid actions per episode.
  - **Efficiency**: steps-to-completion; penalties for unnecessary steps.
  - **Quality**: hallucination rate / faithfulness to evidence (if text-based).

- **Actions for agents:**
  - [ ] Create a small **evaluation set** of 20–50 representative tasks that will not be used for training.
  - [ ] Implement a `metrics` module (e.g. `metrics.py`) that:
    - [ ] Given an episode trajectory, computes all key metrics.
    - [ ] Can be reused by training scripts and evaluation scripts.
  - [ ] Add a short README section describing what “good” means, using these metrics.

---

## Phase 2 — Refine the environment for learnability

- **Goal**: Ensure the environment is **clear, stable, and learnable** for RL.

- **Environment checklist for agents:**
  - [ ] **Observation schema**
    - [ ] Define a typed `Observation` structure (e.g. dataclass / Pydantic model) instead of loose dicts.
    - [ ] Ensure observations expose all information needed to act correctly.
    - [ ] Avoid unnecessary noise or unstable fields (timestamps, random IDs) unless they are essential.
  - [ ] **Action space**
    - [ ] Define an `Action` structure (discrete enum or structured dict) with validation.
    - [ ] Make invalid actions detectable and consistently handled (e.g. special error state + small penalty).
  - [ ] **Episode control**
    - [ ] Verify `reset` is deterministic given a seed.
    - [ ] Confirm terminal conditions (`done`) are correct and not triggered too early or too late.
    - [ ] Add tests that run a fixed-seed episode twice and compare key parts of the trajectory.
  - [ ] **Debuggability**
    - [ ] Implement optional **step tracing** (obs → action → reward → next_obs → done) via a debug flag.
    - [ ] Implement a human-readable `render()` or `pretty_print_episode()` to inspect a full trajectory.

The environment should pass a simple sanity test: a smart human looking at observations and the task should find the correct next action reasonably clear most of the time.

---

## Phase 3 — Reward system redesign

- **Goal**: Make rewards **aligned, dense-enough, and hard to game**.

- **Design principles:**
  - **Good rewards usually:**
    - Give strong positive reward for true success.
    - Give small penalties for invalid, useless, or clearly wrong actions.
    - Use intermediate rewards only when they correlate with final success.
  - **Bad rewards usually:**
    - Reward verbosity instead of correctness.
    - Reward formatting instead of substance.
    - Reward partial-but-wrong behavior.
    - Are very sparse (almost no signal until the very end).

- **Actions for agents:**
  - [ ] Centralize reward logic in a dedicated module (e.g. `reward.py` or `reward_components.py`).
  - [ ] Factor reward into small functions:
    - [ ] `reward_success(episode_state)`
    - [ ] `penalize_invalid(action, state)`
    - [ ] Optional `shaping_reward(transition)` (only if strongly predictive of success).
  - [ ] Implement a single `compute_reward(transition)` entry point that the environment calls.
  - [ ] Add unit tests with **hand-crafted trajectories**:
    - [ ] Confirm that truly successful behavior gets much higher total return than bad behavior.
    - [ ] Confirm that “gaming” strategies (e.g. doing nothing, spamming actions) get low or negative return.
  - [ ] Add a script/notebook that:
    - [ ] Runs a set of successful and failed episodes.
    - [ ] Plots per-step reward and total return for quick visual inspection.

Aim for a clear separation: environment manages state; reward module expresses what the project values.

---

## Phase 4 — Baselines and evaluation harness

- **Goal**: Establish simple baselines and a repeatable way to test agents.

- **Baseline agents:**
  - [ ] Implement a `RandomAgent` (uniform random over valid actions).
  - [ ] Implement one or more `HeuristicAgent`s (rule-based, or a fixed prompt calling an HF model without RL).

- **Evaluation harness:**
  - [ ] Create an `evaluate_agent.py` script that:
    - [ ] Accepts an `Agent` implementation (heuristic, HF model, or RL policy).
    - [ ] Runs N episodes on the held-out evaluation set.
    - [ ] Computes metrics via the shared `metrics` module.
    - [ ] Outputs results in a machine-readable format (JSON / CSV) plus a short human summary.
  - [ ] Ensure this evaluation works:
    - [ ] Against the **local** environment.
    - [ ] Against the **HF-hosted** environment (if applicable), with minimal configuration changes.

The project is ready for serious RL when a trained policy can reliably outperform `RandomAgent` and simple heuristics on the evaluation set.

---

## Phase 5 — Agent interface and training loop

- **Goal**: Make it easy to plug in different agents (HF models, RL policies, heuristics) and train them.

- **Standard `Agent` interface:**
  - [ ] Define an abstract `Agent` class or protocol with:
    - [ ] `reset()` (optional, for per-episode state).
    - [ ] `act(observation, history)` → `Action`.
  - [ ] Implement adapters for:
    - [ ] HF text models (prompt → model → text → parsed `Action`).
    - [ ] RL policies (e.g. neural network policy from an RL library).
    - [ ] Heuristic / rule-based policies.

- **Training harness:**
  - [ ] Create a `training/` package (or similar) to hold:
    - [ ] Experiment configuration (YAML/JSON or CLI args).
    - [ ] An RL loop (custom, TRL, Stable-Baselines, etc.) wired to:
      - [ ] The environment (local or HF-hosted wrapper).
      - [ ] The reward module.
      - [ ] The metrics and logging system.
  - [ ] Add structured logging (e.g. `wandb`, `tensorboard`, HF logs) for:
    - [ ] Episode returns.
    - [ ] Key metrics (success rate, invalid actions, steps).
    - [ ] Example trajectories for inspection.

This phase should end with a simple, documented command like:

```bash
python -m training.run_experiment --config configs/basic_rl.yaml
```

that launches a full training run.

---

## Phase 6 — Curriculum and task diversity

- **Goal**: Make the environment challenging but learnable, and avoid overfitting to narrow patterns.

- **Curriculum:**
  - [ ] Define difficulty tiers for tasks (e.g. `easy`, `medium`, `hard`).
  - [ ] Implement sampling logic that:
    - [ ] Starts training mostly on easy tasks.
    - [ ] Gradually mixes in harder tasks as performance improves.

- **Diversity:**
  - [ ] Ensure the environment can:
    - [ ] Load or generate a diverse set of tasks.
    - [ ] Optionally randomize certain aspects (domain randomization) to improve generalization.

The curriculum logic should be configurable so experiments can compare “no curriculum” vs “curriculum” easily.

---

## Phase 7 — SFT-first, RL-second (if demonstrations exist)

- **Goal**: Use supervised fine-tuning (SFT) to teach the basic behavior, and RL to refine it.

- **Supervised fine-tuning:**
  - [ ] Collect or curate demonstration trajectories (obs → action sequences) that represent good behavior.
  - [ ] Build an SFT dataset from these demonstrations.
  - [ ] Fine-tune the base HF model on this dataset so it learns:
    - [ ] The task format.
    - [ ] Basic strategies and valid action patterns.

- **RL on top of SFT:**
  - [ ] Use the SFT model as the initial policy for RL.
  - [ ] Focus RL optimization on:
    - [ ] Reducing invalid actions.
    - [ ] Increasing success rate on harder tasks.
    - [ ] Improving efficiency (fewer steps).

- **Optional: offline RL / replay:**
  - [ ] Store trajectories (obs, action, reward) in a replay buffer or dataset.
  - [ ] Allow experiments that:
    - [ ] Re-run reward computations with different reward settings.
    - [ ] Use off-policy RL or offline RL algorithms if desired.

---

## Phase 8 — Documentation and agent onboarding

- **Goal**: Make the project easy for new humans and agents (including code assistants) to work on.

- **Documentation tasks:**
  - [ ] Update the main `README` with:
    - [ ] A short description of the environment and its purpose.
    - [ ] How to run a single episode interactively.
    - [ ] How to run evaluation for a given agent.
    - [ ] How to start a basic RL training run.
  - [ ] Add an `ENVIRONMENT_OVERVIEW.md` or expand this file with:
    - [ ] Observation format.
    - [ ] Action format.
    - [ ] Reward components and their rationale.
  - [ ] Provide one or two example notebooks showing:
    - [ ] Manual stepping through a single episode.
    - [ ] Visualizing reward traces and metrics for a few runs.

With these pieces in place, most future work should focus on **iterating environment design, reward shaping, and evaluation**, rather than rewriting the entire project structure. This maximizes the learning signal that the Hugging Face environment can provide to any agent plugged into it.

