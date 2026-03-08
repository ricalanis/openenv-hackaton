#!/usr/bin/env python3
"""Evaluation harness for the FSDS Cleaning Environment.

Runs an agent on the held-out evaluation set and outputs metrics in JSON and a human-readable summary.

Usage:
    python -m fsds_cleaning_env.evaluate_agent --agent heuristic --base-url http://localhost:8000
    python -m fsds_cleaning_env.evaluate_agent --agent random --base-url https://YOUR-SPACE.hf.space --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root for imports when run as script (parent of fsds_cleaning_env)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fsds_cleaning_env.agents import HeuristicAgent, LLMAgent, RandomAgent
from fsds_cleaning_env.client import FSDSCleaningEnv
from fsds_cleaning_env.dataset_generators import EVAL_SEEDS
from fsds_cleaning_env.evaluation_tasks import EVAL_TASKS
from fsds_cleaning_env.metrics import EpisodeMetrics, aggregate_metrics, compute_episode_metrics


def run_evaluation(
    agent,
    base_url: str = "http://localhost:8000",
    tasks: list | None = None,
    max_episodes_per_task: int = 1,
) -> dict:
    """Run agent on evaluation tasks and return results."""
    tasks = tasks or EVAL_TASKS
    all_metrics: list = []
    episode_metrics_list: list = []

    with FSDSCleaningEnv(base_url=base_url).sync() as env:
        for eval_task in tasks:
            seed = EVAL_SEEDS.get(eval_task.task_id, [42])[eval_task.eval_index]
            for ep in range(max_episodes_per_task):
                try:
                    trajectory = agent.run_episode(
                        env,
                        task_id=eval_task.task_id,
                        max_steps=eval_task.max_steps,
                        seed=seed,
                    )
                    metrics = compute_episode_metrics(trajectory)
                    episode_metrics_list.append(metrics)
                    all_metrics.append({
                        "task_name": eval_task.name,
                        "task_id": eval_task.task_id,
                        "episode": ep,
                        "success": metrics.success,
                        "total_return": metrics.total_return,
                        "steps": metrics.steps,
                        "invalid_actions": metrics.invalid_actions,
                        "quality_gate_passed": metrics.quality_gate_passed,
                        "retention_ratio": metrics.retention_ratio,
                    })
                except Exception as e:
                    episode_metrics_list.append(
                        EpisodeMetrics(
                            success=False,
                            total_return=0.0,
                            steps=0,
                            invalid_actions=0,
                            quality_gate_passed=False,
                            retention_ratio=None,
                        )
                    )
                    all_metrics.append({
                        "task_name": eval_task.name,
                        "task_id": eval_task.task_id,
                        "episode": ep,
                        "error": str(e),
                        "success": False,
                        "total_return": 0.0,
                        "steps": 0,
                        "invalid_actions": 0,
                    })

    agg = aggregate_metrics(episode_metrics_list)

    return {
        "agent": getattr(agent, "__class__", type(agent)).__name__,
        "base_url": base_url,
        "n_episodes": len(all_metrics),
        "aggregate": {
            "episodes": agg.episodes,
            "success_rate": agg.success_rate,
            "avg_return": agg.avg_return,
            "avg_steps": agg.avg_steps,
            "avg_invalid_actions": agg.avg_invalid_actions,
        },
        "episodes": all_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an agent on the FSDS Cleaning Environment")
    parser.add_argument(
        "--agent",
        choices=["random", "heuristic", "llm"],
        default="heuristic",
        help="Which agent to evaluate",
    )
    parser.add_argument(
        "--model-path",
        default="./data-cleaning-grpo-final",
        help="Path to trained LLM checkpoint (used when --agent llm)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the environment (local or HF Space)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write results to this JSON file",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=1,
        help="Number of episodes per evaluation task",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for RandomAgent (optional)",
    )
    args = parser.parse_args()

    if args.agent == "random":
        agent = RandomAgent(rng=__import__("random").Random(args.seed) if args.seed else None)
    elif args.agent == "llm":
        agent = LLMAgent(model_path=args.model_path)
    else:
        agent = HeuristicAgent()

    results = run_evaluation(
        agent,
        base_url=args.base_url,
        max_episodes_per_task=args.episodes_per_task,
    )

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))

    agg = results["aggregate"]
    print(f"\n=== Evaluation: {results['agent']} on {args.base_url} ===")
    print(f"Episodes: {agg['episodes']}")
    print(f"Success rate: {agg['success_rate']:.2%}")
    print(f"Avg return: {agg['avg_return']:.4f}")
    print(f"Avg steps: {agg['avg_steps']:.1f}")
    print(f"Avg invalid actions: {agg['avg_invalid_actions']:.2f}")
    if args.output:
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
