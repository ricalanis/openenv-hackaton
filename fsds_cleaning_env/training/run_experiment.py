#!/usr/bin/env python3
"""Run a training experiment with the FSDS Cleaning Environment.

Usage:
    python -m fsds_cleaning_env.training.run_experiment --config configs/basic_rl.json
    python -m fsds_cleaning_env.training.run_experiment --config configs/basic_rl.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Ensure project root is in path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fsds_cleaning_env.agents import HeuristicAgent, RandomAgent
from fsds_cleaning_env.client import FSDSCleaningEnv
from fsds_cleaning_env.curriculum import ALL_TASK_IDS, CurriculumScheduler
from fsds_cleaning_env.metrics import EpisodeMetrics, aggregate_metrics, compute_episode_metrics
from fsds_cleaning_env.training.config import ExperimentConfig


def _get_agent(config: ExperimentConfig):
    if config.agent == "random":
        rng = random.Random(config.seed) if config.seed is not None else None
        return RandomAgent(rng=rng)
    if config.agent == "heuristic":
        return HeuristicAgent()
    raise ValueError(f"Unknown agent: {config.agent}")


def run_experiment(config: ExperimentConfig) -> dict:
    """Run n_episodes and return aggregated results + per-episode metrics.

    When ``config.curriculum`` is True the experiment uses a
    ``CurriculumScheduler`` to drive task selection and noise level instead
    of a fixed ``task_id``.  Difficulty promotes automatically once the
    rolling success-rate window crosses the threshold for the current stage.
    """
    agent = _get_agent(config)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    episode_metrics_list: list[EpisodeMetrics] = []
    trajectories_log: list[dict] = []

    # Build curriculum scheduler when requested.
    scheduler: CurriculumScheduler | None = None
    if config.curriculum:
        task_ids = config.curriculum_task_ids or list(ALL_TASK_IDS)
        scheduler = CurriculumScheduler(
            task_ids=task_ids,
            mode=config.curriculum_mode,
            start_level=config.curriculum_start_level,
            rng=random.Random(config.seed) if config.seed is not None else None,
        )

    with FSDSCleaningEnv(base_url=config.base_url).sync() as env:
        for ep in range(config.n_episodes):
            seed = config.seed + ep if config.seed is not None else None

            if scheduler is not None:
                cur_task = scheduler.next_task(seed=seed)
                task_id = cur_task.task_id
                max_steps = cur_task.max_steps
                extra_kwargs = {
                    "dataset_n_rows": cur_task.n_rows,
                    "noise_profile_override": cur_task.noise_profile,
                }
                difficulty = cur_task.difficulty
            else:
                task_id = config.task_id
                max_steps = config.max_steps_per_episode
                extra_kwargs = {}
                difficulty = "n/a"

            try:
                trajectory = agent.run_episode(
                    env,
                    task_id=task_id,
                    max_steps=max_steps,
                    seed=seed,
                    **extra_kwargs,
                )
                metrics = compute_episode_metrics(trajectory)
                episode_metrics_list.append(metrics)
                ep_log: dict = {
                    "episode": ep,
                    "task_id": task_id,
                    "difficulty": difficulty,
                    "success": metrics.success,
                    "total_return": metrics.total_return,
                    "steps": metrics.steps,
                    "invalid_actions": metrics.invalid_actions,
                }
                if scheduler is not None:
                    promoted = scheduler.record_episode(metrics.success)
                    ep_log["promoted_to"] = scheduler.level_name if promoted else None
            except Exception as e:
                metrics = EpisodeMetrics(
                    success=False,
                    total_return=0.0,
                    steps=0,
                    invalid_actions=0,
                    quality_gate_passed=False,
                    retention_ratio=None,
                )
                episode_metrics_list.append(metrics)
                ep_log = {
                    "episode": ep,
                    "task_id": task_id,
                    "difficulty": difficulty,
                    "error": str(e),
                    "success": False,
                    "total_return": 0.0,
                    "steps": 0,
                    "invalid_actions": 0,
                }
                if scheduler is not None:
                    scheduler.record_episode(False)
                    ep_log["promoted_to"] = None

            trajectories_log.append(ep_log)

            if (ep + 1) % config.log_interval == 0:
                agg = aggregate_metrics(episode_metrics_list)
                sched_info = (
                    f" difficulty={scheduler.level_name}" if scheduler else ""
                )
                print(
                    f"[{ep + 1}/{config.n_episodes}]{sched_info} "
                    f"success_rate={agg.success_rate:.2%} "
                    f"avg_return={agg.avg_return:.4f} "
                    f"avg_steps={agg.avg_steps:.1f} "
                    f"avg_invalid={agg.avg_invalid_actions:.2f}"
                )

    agg = aggregate_metrics(episode_metrics_list)
    results: dict = {
        "run_id": run_id,
        "config": asdict(config),
        "aggregate": {
            "episodes": agg.episodes,
            "success_rate": agg.success_rate,
            "avg_return": agg.avg_return,
            "avg_steps": agg.avg_steps,
            "avg_invalid_actions": agg.avg_invalid_actions,
        },
        "episodes": trajectories_log,
    }
    if scheduler is not None:
        results["curriculum"] = scheduler.summary()

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"run_{run_id}.json"
    out_file.write_text(json.dumps(results, indent=2))

    # Log summary to log_dir
    summary_file = log_dir / f"summary_{run_id}.json"
    summary_file.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "aggregate": results["aggregate"],
                "config": results["config"],
                **({"curriculum": results["curriculum"]} if scheduler is not None else {}),
            },
            indent=2,
        )
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FSDS Cleaning Environment training experiment")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to config file (JSON or YAML)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try cwd first, then package-relative (fsds_cleaning_env/configs/)
        if not config_path.exists():
            pkg_root = Path(__file__).resolve().parent.parent
            alt = pkg_root / config_path
            if alt.exists():
                config_path = alt
    config = ExperimentConfig.from_file(config_path)

    print(f"Starting experiment: agent={config.agent} task={config.task_id} n_episodes={config.n_episodes}")
    results = run_experiment(config)

    print(f"\n=== Experiment complete ===")
    print(f"Success rate: {results['aggregate']['success_rate']:.2%}")
    print(f"Avg return: {results['aggregate']['avg_return']:.4f}")
    print(f"Avg steps: {results['aggregate']['avg_steps']:.1f}")
    print(f"Results saved to: {config.output_dir}/run_{results['run_id']}.json")


if __name__ == "__main__":
    main()
