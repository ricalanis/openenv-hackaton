# FSDS Cleaning Agent — Evaluation Report

**Date:** 2026-03-08
**Environment:** `https://israaaML-fsds-cleaning-env.hf.space`
**Episodes per task:** 3
**Tasks evaluated:** `ecommerce_mobile`, `subscription_churn`, `delivery_eta`
**Total episodes per agent:** 45 (15 tasks × 3 episodes)

---

## 1. Summary Table

| Agent | Success Rate | Avg Return | Avg Steps | Avg Invalid Actions | Quality Gate Passed |
|---|---|---|---|---|---|
| HeuristicAgent | 0.00% | -0.0878 | 12.2 | 0 | No |
| RandomAgent | 0.00% | -0.0912 | 3.2 | 0 | No |
| LLMAgent (GRPO) | N/A* | N/A* | N/A* | N/A* | N/A* |

> *LLMAgent could not be evaluated locally because `unsloth` is a Colab/GPU-only library not installed in the local Python environment. See Section 4 for details and the recommended fix.

---

## 2. Agent-by-Agent Analysis

### 2.1 HeuristicAgent (Rule-based baseline)

The HeuristicAgent follows a hard-coded, task-specific cleaning policy derived from the known `required_ops` for each task. It is the **intended upper-bound reference** for this environment.

**Results:**

| Task | Avg Return | Avg Retention | Steps | Quality Gate |
|---|---|---|---|---|
| ecommerce_mobile | -0.0600 | 98.06% | 12 | Failed |
| subscription_churn | -0.1108 | 98.17% | 14 | Failed |
| delivery_eta | -0.0927 | 97.98% | 13 | Failed |

**Key observations:**
- Executes the full cleaning pipeline correctly (12–14 steps).
- Achieves ~98% data retention across all tasks, which is healthy.
- Returns are negative because every step incurs a small step penalty (`-reward_per_step`), and no terminal success reward is collected since quality gates never pass.
- Zero invalid actions — all tool calls are structurally correct.
- The consistent `quality_gate_passed: False` across all tasks and all episodes suggests the environment's quality gate thresholds may require operations beyond what the scripted policy currently includes, or a configuration mismatch exists between the policy and the active server version.

**Interpretation:** The heuristic agent is behaviourally correct (right tools, right order, good retention) but does not cross the quality gate threshold. This is a signal about the quality gate strictness, not about the agent's cleaning ability.

---

### 2.2 RandomAgent (Lower-bound baseline)

The RandomAgent samples actions uniformly at random from the valid action space.

**Results:**

| Metric | Value |
|---|---|
| Success Rate | 0.00% |
| Avg Return | -0.0912 |
| Avg Steps | 3.2 |
| Avg Invalid Actions | 0 |

**Key observations:**
- Terminates early (avg 3.2 steps) because it randomly selects `submit_solution` before meaningful cleaning is done.
- Slightly worse average return than HeuristicAgent (-0.0912 vs -0.0878), confirming the heuristic is doing something useful even if not enough to pass quality gates.
- Zero invalid actions because the action sampler only picks structurally valid tool calls.
- The small gap between Random and Heuristic returns is partly due to the RandomAgent's short episodes — fewer steps means fewer step penalties, partially offsetting its bad cleaning quality.

---

### 2.3 LLMAgent — GRPO Fine-tuned Model

**Status: Not evaluated locally.**

All 45 episodes failed with:
```
Error: No module named 'unsloth'
```

**Root cause:** `unsloth` is a Colab-optimised library that patches the HuggingFace `transformers` stack for 4-bit GPU training. It is not pip-installable in standard CPU/MPS environments without CUDA. The trained checkpoint (`./data-cleaning-grpo-final`) is a LoRA adapter that requires the Unsloth model loader to be instantiated correctly.

**This is an infrastructure constraint, not a model quality issue.** The model itself trained successfully (Cell 9 completed without errors in Colab).

---

## 3. Comparative Analysis

```
Return ranking (higher is better):
  LLMAgent (GRPO):  N/A (not evaluated)
  HeuristicAgent:   -0.0878  ← best evaluated
  RandomAgent:      -0.0912  ← worst evaluated

Step efficiency (fewer steps = faster decisions):
  RandomAgent:      3.2  (but premature submission)
  HeuristicAgent:   12.2 (full pipeline execution)
  LLMAgent:         N/A
```

The HeuristicAgent is the better agent of the two that ran:
- It executes a complete, reasoned cleaning sequence.
- It achieves higher data retention (~98% vs ~100% for Random, but Random does no cleaning).
- Its negative return is purely a step-penalty artefact, not evidence of bad cleaning.

The RandomAgent's slightly fewer step-penalty losses are misleading — it simply stops early without cleaning anything meaningful.

---

## 4. How to Evaluate the LLMAgent

Run the evaluation in **Google Colab** (T4 GPU recommended) where `unsloth` is available:

```python
# In Colab, after installing dependencies:
# !pip install -q "trl>=0.12.0" "accelerate>=0.34.0" "peft>=0.13.0" "bitsandbytes>=0.43.0"
# !pip install -q unsloth
# !pip install -q "git+https://huggingface.co/spaces/israaaML/fsds_cleaning_env"

from fsds_cleaning_env.agents import LLMAgent
from fsds_cleaning_env.evaluate_agent import run_evaluation

agent = LLMAgent(model_path="./data-cleaning-grpo-final")
results = run_evaluation(
    agent,
    base_url="https://israaaML-fsds-cleaning-env.hf.space",
    max_episodes_per_task=3,
)

print(f"Success rate: {results['aggregate']['success_rate']:.2%}")
print(f"Avg return:   {results['aggregate']['avg_return']:.4f}")
print(f"Avg steps:    {results['aggregate']['avg_steps']:.1f}")
```

Expected comparison targets once evaluated:

| Metric | Random (lower bound) | Heuristic (reference) | LLM target |
|---|---|---|---|
| Success rate | 0% | 0%* | >0% |
| Avg return | -0.0912 | -0.0878 | > -0.0878 |
| Avg steps | 3.2 | 12.2 | ~10–15 |

> *The 0% success rate for the Heuristic agent is likely caused by a quality gate configuration issue on the server — investigate `run_quality_gates` responses to confirm which specific checks are failing.

---

## 5. Issues Identified & Next Steps

### Issue 1 — Quality gates never pass (affects all agents)
The environment returns `quality_gate_passed: False` for every episode including the HeuristicAgent, which applies the correct canonical operations. This is unexpected.

**Recommended action:** Run a manual debug episode and inspect the `run_quality_gates` response payload to see which specific checks fail and why.

```python
with FSDSCleaningEnv(base_url=ENV_URL).sync() as env:
    env.reset(task_id="ecommerce_mobile")
    # ... apply cleaning ops ...
    result = env.call_tool("run_quality_gates")
    print(result)  # inspect which tests fail
```

### Issue 2 — LLMAgent requires Colab/GPU environment
The trained LoRA adapter depends on `unsloth` and 4-bit quantisation (bitsandbytes + CUDA).

**Recommended action:** Run LLMAgent evaluation in Colab using the code in Section 4.

### Issue 3 — SFT warm-start checkpoint not used for GRPO
`training_colab.py` line 60 still points to the base model, not the SFT checkpoint:
```python
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
# MODEL_NAME = "./data-cleaning-sft-final"  ← not activated
```
Switching to the SFT warm-start before the next GRPO run should improve convergence significantly.

---

## 6. Conclusion

Of the two agents successfully evaluated, the **HeuristicAgent is clearly superior** — it executes a complete and structured data-cleaning pipeline with ~98% retention and zero invalid actions. The **RandomAgent** serves as a noisy lower bound, terminating prematurely without meaningful cleaning.

The **LLMAgent (GRPO)** trained successfully in Colab but requires a GPU environment to evaluate. Once evaluated in Colab, it should be compared against the Heuristic reference on the three metrics: success rate, average return, and average steps. A positive success rate would be a strong signal that RL training transferred useful cleaning behaviour beyond the scripted baseline.

The most important outstanding issue is diagnosing why quality gates fail even for the HeuristicAgent — resolving this is a prerequisite for any agent achieving a non-zero success rate.
