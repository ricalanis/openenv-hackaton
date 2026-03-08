"""
Agent Comparison — FSDS Cleaning Environment
=============================================
Benchmarks four agents on the held-out evaluation set and prints a
side-by-side comparison table.

Agents evaluated
----------------
  1. RandomAgent    — lower bound (uniform random actions)
  2. HeuristicAgent — upper bound (scripted oracle policy)
  3. SFT model      — supervised fine-tuned checkpoint warm-start
  4. GRPO model     — RL-trained checkpoint (SFT → GRPO)

Run in Colab after both training files have completed:
  - training_sft.py  → ./data-cleaning-sft-final
  - training_colab.py → ./data-cleaning-grpo-final
"""

# ── Cell 1 ▸ Install (skip if already installed) ──────────────────────
# %%
# !pip install -q "openenv-core[core]>=0.2.1"
# !pip install -q "git+https://huggingface.co/spaces/israaaML/fsds_cleaning_env"
# !pip uninstall -y vllm
# !pip install -q unsloth


# ── Cell 2 ▸ Imports & config ─────────────────────────────────────────
# %%
import json
from pathlib import Path

ENV_URL          = "https://israaaML-fsds-cleaning-env.hf.space"
SFT_MODEL_PATH   = "./data-cleaning-sft-final"
GRPO_MODEL_PATH  = "./data-cleaning-grpo-final"
EPISODES_PER_TASK = 3          # increase for more reliable estimates (slower)
OUTPUT_FILE      = "./results_comparison.json"


# ── Cell 3 ▸ Connect to environment & sanity check ────────────────────
# %%
from fsds_cleaning_env import FSDSCleaningEnv
from fsds_cleaning_env.evaluation_tasks import EVAL_TASKS
from fsds_cleaning_env.evaluate_agent import run_evaluation
from fsds_cleaning_env.agents import RandomAgent, HeuristicAgent, LLMAgent
from fsds_cleaning_env.metrics import aggregate_metrics, compute_episode_metrics

with FSDSCleaningEnv(base_url=ENV_URL).sync() as env:
    env.reset(task_id="ecommerce_mobile")
    brief = env.call_tool("get_task_brief")
    print(f"Connected to env. Task: {brief.get('title')}")
    tasks_list = env.call_tool("list_tasks")
    print(f"Available tasks: {[t['task_id'] for t in tasks_list.get('tasks', [])]}")
print(f"\nEval tasks ({len(EVAL_TASKS)} scenarios):")
for t in EVAL_TASKS:
    print(f"  {t.name}  (task_id={t.task_id}, seed_index={t.eval_index})")


# ── Cell 4 ▸ Run all agents ───────────────────────────────────────────
# %%
results = {}

# ── 4a. Random (lower bound) ──────────────────────────────────────────
print("\n[1/4] Evaluating RandomAgent …")
results["random"] = run_evaluation(
    RandomAgent(),
    base_url=ENV_URL,
    max_episodes_per_task=EPISODES_PER_TASK,
)
agg = results["random"]["aggregate"]
print(f"  success={agg['success_rate']:.0%}  return={agg['avg_return']:.4f}  steps={agg['avg_steps']:.1f}")

# ── 4b. Heuristic (upper bound) ───────────────────────────────────────
print("\n[2/4] Evaluating HeuristicAgent …")
results["heuristic"] = run_evaluation(
    HeuristicAgent(),
    base_url=ENV_URL,
    max_episodes_per_task=EPISODES_PER_TASK,
)
agg = results["heuristic"]["aggregate"]
print(f"  success={agg['success_rate']:.0%}  return={agg['avg_return']:.4f}  steps={agg['avg_steps']:.1f}")

# ── 4c. SFT model ─────────────────────────────────────────────────────
print(f"\n[3/4] Evaluating SFT model ({SFT_MODEL_PATH}) …")
results["sft"] = run_evaluation(
    LLMAgent(model_path=SFT_MODEL_PATH, temperature=0.0),
    base_url=ENV_URL,
    max_episodes_per_task=EPISODES_PER_TASK,
)
agg = results["sft"]["aggregate"]
print(f"  success={agg['success_rate']:.0%}  return={agg['avg_return']:.4f}  steps={agg['avg_steps']:.1f}")

# ── 4d. GRPO model ────────────────────────────────────────────────────
print(f"\n[4/4] Evaluating GRPO model ({GRPO_MODEL_PATH}) …")
results["grpo"] = run_evaluation(
    LLMAgent(model_path=GRPO_MODEL_PATH, temperature=0.0),
    base_url=ENV_URL,
    max_episodes_per_task=EPISODES_PER_TASK,
)
agg = results["grpo"]["aggregate"]
print(f"  success={agg['success_rate']:.0%}  return={agg['avg_return']:.4f}  steps={agg['avg_steps']:.1f}")


# ── Cell 5 ▸ Comparison table ─────────────────────────────────────────
# %%
AGENTS = [
    ("Random",    "random"),
    ("Heuristic", "heuristic"),
    ("SFT",       "sft"),
    ("GRPO",      "grpo"),
]

COL_W = 12

def _col(v, w=COL_W):
    return str(v).center(w)

header = (
    f"{'Agent':<14}"
    + _col("Success %")
    + _col("Avg Return")
    + _col("Avg Steps")
    + _col("Avg Invalid")
    + _col("Episodes")
)
sep = "-" * len(header)

print("\n" + sep)
print("  FSDS Cleaning Agent Benchmark")
print(sep)
print(header)
print(sep)

for label, key in AGENTS:
    if key not in results:
        continue
    agg = results[key]["aggregate"]
    print(
        f"{label:<14}"
        + _col(f"{agg['success_rate']:.1%}")
        + _col(f"{agg['avg_return']:.4f}")
        + _col(f"{agg['avg_steps']:.1f}")
        + _col(f"{agg['avg_invalid_actions']:.2f}")
        + _col(agg["episodes"])
    )

print(sep)

# Improvement of GRPO over SFT
if "sft" in results and "grpo" in results:
    sft_sr  = results["sft"]["aggregate"]["success_rate"]
    grpo_sr = results["grpo"]["aggregate"]["success_rate"]
    sft_ret  = results["sft"]["aggregate"]["avg_return"]
    grpo_ret = results["grpo"]["aggregate"]["avg_return"]
    print(f"\nGRPO vs SFT — success rate delta : {grpo_sr - sft_sr:+.1%}")
    print(f"GRPO vs SFT — avg return delta   : {grpo_ret - sft_ret:+.4f}")


# ── Cell 6 ▸ Per-task breakdown ───────────────────────────────────────
# %%
# Group per-episode results by task_id for a fine-grained breakdown.
from collections import defaultdict

print("\n=== Per-task success rates ===")
task_ids = sorted({ep["task_id"] for ep in results["heuristic"]["episodes"]})

col_labels = [label for label, _ in AGENTS if _ in results]
keys       = [key   for _, key in AGENTS if key in results]

# Header
print(f"\n{'Task':<30}" + "".join(f"{lbl:>12}" for lbl in col_labels))
print("-" * (30 + 12 * len(col_labels)))

for tid in task_ids:
    row = f"{tid:<30}"
    for key in keys:
        eps = [e for e in results[key]["episodes"] if e["task_id"] == tid]
        if not eps:
            row += f"{'N/A':>12}"
        else:
            sr = sum(1 for e in eps if e.get("success", False)) / len(eps)
            row += f"{sr:>11.0%} "
    print(row)


# ── Cell 7 ▸ Save results ─────────────────────────────────────────────
# %%
Path(OUTPUT_FILE).write_text(json.dumps(results, indent=2))
print(f"\nFull results saved to {OUTPUT_FILE}")
