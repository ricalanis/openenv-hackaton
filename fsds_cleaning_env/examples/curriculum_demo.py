#!/usr/bin/env python3
"""Curriculum demo — shows how the CurriculumScheduler drives progressive difficulty.

Simulates 60 episodes with a HeuristicAgent (no live server needed for the
scheduling logic demonstration) and prints difficulty promotions as they occur.

To run against a live server (optional — uncomment the env block):
    python examples/curriculum_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util as _ilu, types as _types

# Load curriculum.py directly by path to avoid triggering openenv in __init__.py
_CURRICULUM_FILE = _ROOT / "fsds_cleaning_env" / "curriculum.py"
_spec = _ilu.spec_from_file_location("fsds_cleaning_env.curriculum", _CURRICULUM_FILE)
_curriculum = _types.ModuleType("fsds_cleaning_env.curriculum")
_spec.loader.exec_module(_curriculum)  # type: ignore[union-attr]

ALL_TASK_IDS = _curriculum.ALL_TASK_IDS
CurriculumScheduler = _curriculum.CurriculumScheduler
DIFFICULTY_LEVELS = _curriculum.DIFFICULTY_LEVELS


# ── Offline simulation (no server) ────────────────────────────────────────────

def simulate_curriculum_offline(n_episodes: int = 60) -> None:
    """Walk through curriculum scheduling logic without connecting to a server.

    The HeuristicAgent always succeeds on clean tasks, so the scheduler will
    promote from easy → medium → hard as thresholds are crossed.  We fake the
    success signal based on the difficulty level to illustrate demotion behavior
    at higher difficulties.
    """
    print("=== Curriculum Scheduler — offline simulation ===\n")
    print(f"Levels: {[l.name for l in DIFFICULTY_LEVELS]}")
    print(f"Tasks:  {ALL_TASK_IDS}\n")

    scheduler = CurriculumScheduler(
        task_ids=ALL_TASK_IDS,
        mode="round_robin",
        start_level="easy",
    )

    import random
    rng = random.Random(0)

    for ep in range(n_episodes):
        cur = scheduler.next_task(seed=ep)

        # Simulate success probability: easy=0.9, medium=0.75, hard=0.55
        success_prob = {"easy": 0.90, "medium": 0.75, "hard": 0.55}[cur.difficulty]
        success = rng.random() < success_prob

        promoted = scheduler.record_episode(success)

        status = "[PROMOTED]" if promoted else ""
        print(
            f"  ep={ep:03d} | {cur.difficulty:6s} | {cur.task_id:25s} | "
            f"rows={cur.n_rows:4d} | steps={cur.max_steps:2d} | "
            f"success={str(success):5s} {status}"
        )

    print("\n=== Final scheduler summary ===")
    print(json.dumps(scheduler.summary(), indent=2))


# ── Live run (requires local server) ──────────────────────────────────────────

def run_curriculum_live(
    base_url: str = "http://localhost:8000",
    n_episodes: int = 30,
    seed: int = 0,
) -> None:
    """Run curriculum training against a live environment server."""
    # Lazy imports — openenv and agents only needed for live runs.
    from fsds_cleaning_env.client import FSDSCleaningEnv  # noqa: PLC0415
    from fsds_cleaning_env.agents import HeuristicAgent  # noqa: PLC0415
    from fsds_cleaning_env.metrics import aggregate_metrics, compute_episode_metrics  # noqa: PLC0415

    print(f"=== Curriculum live run ({base_url}) ===\n")

    scheduler = CurriculumScheduler(
        task_ids=ALL_TASK_IDS,
        mode="round_robin",
        start_level="easy",
    )
    agent = HeuristicAgent()
    all_metrics = []

    with FSDSCleaningEnv(base_url=base_url).sync() as env:
        for ep in range(n_episodes):
            cur = scheduler.next_task(seed=seed + ep)
            try:
                trajectory = agent.run_episode(
                    env,
                    task_id=cur.task_id,
                    max_steps=cur.max_steps,
                    seed=seed + ep,
                    dataset_n_rows=cur.n_rows,
                    noise_profile_override=cur.noise_profile,
                )
                m = compute_episode_metrics(trajectory)
                all_metrics.append(m)
                promoted = scheduler.record_episode(m.success)
                print(
                    f"  ep={ep:02d} | {cur.difficulty:6s} | {cur.task_id:25s} | "
                    f"return={m.total_return:.3f} | success={m.success}"
                    + (" [PROMOTED]" if promoted else "")
                )
            except Exception as exc:
                print(f"  ep={ep:02d} | ERROR: {exc}")
                scheduler.record_episode(False)

    if all_metrics:
        agg = aggregate_metrics(all_metrics)
        print(f"\nSuccess rate : {agg.success_rate:.2%}")
        print(f"Avg return   : {agg.avg_return:.4f}")
    print("\nScheduler summary:")
    print(json.dumps(scheduler.summary(), indent=2))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Curriculum demo")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Connect to a live server instead of simulating offline.",
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--n-episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.live:
        run_curriculum_live(
            base_url=args.base_url,
            n_episodes=args.n_episodes,
            seed=args.seed,
        )
    else:
        simulate_curriculum_offline(n_episodes=args.n_episodes)
