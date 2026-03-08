from __future__ import annotations

from dataclasses import dataclass
from typing import List

from fsds_cleaning_env.dataset_generators import EVAL_SEEDS, SIZE_MEDIUM


@dataclass(frozen=True)
class EvaluationTask:
    """Specification for a single evaluation scenario.

    These tasks are meant to be used by evaluation / benchmarking scripts and
    SHOULD NOT be used for on-policy RL training so that evaluation remains
    held-out.

    eval_index selects which fixed seed from EVAL_SEEDS to use, so the same
    table is produced every time for reproducible evaluation.
    """

    name: str
    task_id: str
    description: str
    max_steps: int
    eval_index: int = 0
    n_rows: int = SIZE_MEDIUM


def _expand_eval_tasks() -> List[EvaluationTask]:
    """Build evaluation tasks from base scenarios and fixed seeds."""
    base = [
        ("ecommerce_mobile_baseline", "ecommerce_mobile", "Canonical mobile conversion cleaning task."),
        ("subscription_churn_baseline", "subscription_churn", "Subscription churn table cleaning for churn modeling."),
        ("delivery_eta_baseline", "delivery_eta", "Last-mile delivery ETA cleaning task."),
    ]
    tasks = []
    for name, task_id, desc in base:
        seeds = EVAL_SEEDS.get(task_id, [42])
        for idx, _ in enumerate(seeds):
            tasks.append(
                EvaluationTask(
                    name=f"{name}_seed{idx}",
                    task_id=task_id,
                    description=desc,
                    max_steps=18,
                    eval_index=idx,
                    n_rows=SIZE_MEDIUM,
                )
            )
    return tasks


EVAL_TASKS: List[EvaluationTask] = _expand_eval_tasks()

"""
NOTE FOR FUTURE AGENTS:

- Each EvaluationTask has eval_index pointing to a fixed seed in EVAL_SEEDS.
  Use get_eval_dataset(task_id, eval_index) to get the held-out table.
- For evaluation, call env.reset(task_id=..., seed=EVAL_SEEDS[task_id][eval_index])
  so the environment produces the same table each run.
- To add more tasks: extend EVAL_SEEDS in dataset_generators.py and re-run
  _expand_eval_tasks(), or append manual EvaluationTask entries.
"""

__all__ = ["EvaluationTask", "EVAL_TASKS"]

