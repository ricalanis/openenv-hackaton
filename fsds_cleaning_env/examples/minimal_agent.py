#!/usr/bin/env python3
"""Minimal working agent — the simplest way to run an episode.

This script is the recommended starting point for anyone new to the
FSDS Cleaning Environment.  It shows every step of a complete episode with
inline comments explaining what is happening and why.

Run it against a local server:
    uvicorn fsds_cleaning_env.server.app:app --port 8000 &
    python examples/minimal_agent.py

Or against the HF Space:
    python examples/minimal_agent.py --base-url https://israaaML-fsds-cleaning-env.hf.space

For the full agent guide, see AGENT_GUIDE.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Make sure the package root is on the path ─────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fsds_cleaning_env.client import FSDSCleaningEnv


def run_minimal_episode(base_url: str, task_id: str) -> dict:
    """Run one complete episode with a hand-written cleaning policy.

    Returns the final result dict from submit_solution.
    """
    # ── Connect ───────────────────────────────────────────────────────────────
    # FSDSCleaningEnv is a thin HTTP client.  The `.sync()` context manager
    # opens a connection and closes it cleanly when the block exits.
    with FSDSCleaningEnv(base_url=base_url).sync() as env:

        # ── Reset ─────────────────────────────────────────────────────────────
        # reset() starts a new episode.
        # seed=None → a fresh random table is generated each time (good for training).
        # Use a fixed integer seed for reproducible evaluation.
        obs = env.reset(task_id=task_id, seed=None)
        print(f"\n{'='*60}")
        print(f"Episode started: task={obs['task_id']}  max_steps={obs['max_steps']}")
        print(f"{'='*60}\n")

        # ── Step 1: Understand the task ───────────────────────────────────────
        # get_task_brief tells us the objective, the target column we must not
        # touch, and the list of required operations we need to apply.
        brief = env.call_tool("get_task_brief")
        print("Task:", brief.get("title"))
        print("Objective:", brief.get("objective", "")[:120])
        required_ops = brief.get("required_ops", [])
        print(f"Required operations ({len(required_ops)}):")
        for op in required_ops:
            col = op.get("column", "(no column)")
            print(f"  • {op['operation']} → {col}")

        # ── Step 2: Profile the data ──────────────────────────────────────────
        # profile_data returns shape, dtypes, missing counts, duplicate count,
        # and invalid token counts.  This is crucial for deciding what to clean.
        profile = env.call_tool("profile_data")
        print(f"\nShape: {profile.get('shape')}")
        print(f"Duplicates: {profile.get('n_duplicates', 0)}")
        missing = profile.get("missing_counts", {})
        print("Missing counts:", {k: v for k, v in missing.items() if v > 0})

        total_reward = 0.0

        # ── Step 3: Apply cleaning operations ────────────────────────────────
        # We follow the required_ops list from the task brief.
        # A real LLM agent would decide these dynamically from the profile;
        # here we iterate the brief's list directly to keep things simple.
        for op_spec in required_ops:
            operation = op_spec["operation"]
            column = op_spec.get("column")

            kwargs: dict = {"operation": operation}
            if column:
                kwargs["column"] = column

            result = env.call_tool("apply_cleaning_operation", **kwargs)
            reward = float(result.get("reward", 0.0))
            total_reward += reward

            status = result.get("status", "?")
            score = result.get("quality_score", "?")
            delta = result.get("quality_delta", "?")
            col_label = f" [{column}]" if column else ""
            print(f"  {operation}{col_label}: status={status}  Δquality={delta}  reward={reward:+.3f}")

            # Stop early if the environment signals the episode is done
            # (e.g. step budget exhausted).
            if result.get("done"):
                print("Episode ended early (step budget).")
                return result

        # ── Step 4: Run quality gates ─────────────────────────────────────────
        # Quality gates verify: no missing values, no duplicates, target column
        # preserved, row retention above threshold, dtype alignment, stability.
        # Run this BEFORE submitting — it adds a +0.15 bonus if all pass, and
        # gives you a chance to fix any failures before the final submission.
        gate_result = env.call_tool("run_quality_gates")
        gate_reward = float(gate_result.get("reward", 0.0))
        total_reward += gate_reward
        passed = gate_result.get("passed", False)
        print(f"\nQuality gates: {'PASSED ✓' if passed else 'FAILED ✗'}  reward={gate_reward:+.3f}")

        if not passed:
            # Print which gate failed so you know what to fix.
            for gate_name, gdata in gate_result.get("gate_results", {}).items():
                if not gdata.get("passed", True):
                    print(f"  FAILED: {gate_name} — {gdata.get('details', '')}")

        # ── Step 5: Submit ────────────────────────────────────────────────────
        # submit_solution ends the episode and returns the final_reward.
        # final_reward = 0.45 × quality_score
        #              + 0.30 × gate_passed
        #              + 0.25 × required_op_coverage
        final = env.call_tool("submit_solution")
        final_reward = float(final.get("final_reward", 0.0))
        total_reward += final_reward

        print(f"\nFinal reward:  {final_reward:.4f}")
        print(f"Quality score: {final.get('quality_score', '?')}")
        print(f"Gate passed:   {final.get('gate_passed', '?')}")
        print(f"Op coverage:   {final.get('required_op_coverage', '?')}")
        print(f"Cumulative:    {total_reward:.4f}")
        print(f"Success:       {'YES' if final_reward > 0.5 else 'NO'}")

        return final


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal FSDS Cleaning Agent")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Environment server URL",
    )
    parser.add_argument(
        "--task",
        default="ecommerce_mobile",
        choices=["ecommerce_mobile", "subscription_churn", "delivery_eta"],
        help="Task ID to run",
    )
    args = parser.parse_args()

    result = run_minimal_episode(base_url=args.base_url, task_id=args.task)
    print("\nFull result JSON:")
    print(json.dumps({k: v for k, v in result.items() if k != "operation_log"}, indent=2))


if __name__ == "__main__":
    main()
