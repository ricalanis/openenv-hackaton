"""Curriculum scheduler for progressive difficulty training.

Manages task selection and difficulty progression for RL training.
The scheduler promotes agents from easier to harder episodes as performance
improves, ensuring stable gradient signals early in training without
sacrificing challenge once the agent becomes capable.

Difficulty progression:
    easy   -> light noise, 300 rows, 22 steps
    medium -> medium noise, 500 rows, 18 steps
    hard   -> heavy noise, 1000 rows, 15 steps

Task sampling modes:
    round_robin  - cycle through task_ids in order (default)
    random       - uniform random pick at each episode
    single       - stick to one task_id (pass task_ids as a 1-element list)
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional

from fsds_cleaning_env.dataset_generators import (
    SIZE_LARGE,
    SIZE_MEDIUM,
    SIZE_SMALL,
    NoiseProfile,
)


# ── Difficulty definitions ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class DifficultyLevel:
    """Parameters for one curriculum stage."""

    name: str
    noise_profile: NoiseProfile
    n_rows: int
    max_steps: int
    # Minimum success_rate over `window_size` recent episodes needed to advance.
    promotion_threshold: float
    # How many recent episodes to consider for promotion.
    window_size: int


DIFFICULTY_LEVELS: List[DifficultyLevel] = [
    DifficultyLevel(
        name="easy",
        noise_profile=NoiseProfile.light(),
        n_rows=SIZE_SMALL,
        max_steps=22,
        promotion_threshold=0.70,
        window_size=10,
    ),
    DifficultyLevel(
        name="medium",
        noise_profile=NoiseProfile.medium(),
        n_rows=SIZE_MEDIUM,
        max_steps=18,
        promotion_threshold=0.65,
        window_size=15,
    ),
    DifficultyLevel(
        name="hard",
        noise_profile=NoiseProfile.heavy(),
        n_rows=SIZE_LARGE,
        max_steps=15,
        promotion_threshold=1.01,  # unreachable; "hard" is the final level
        window_size=20,
    ),
]

# Map name to level for convenience.
LEVELS_BY_NAME: dict[str, DifficultyLevel] = {lvl.name: lvl for lvl in DIFFICULTY_LEVELS}

ALL_TASK_IDS: List[str] = ["ecommerce_mobile", "subscription_churn", "delivery_eta"]


# ── CurriculumTask ─────────────────────────────────────────────────────────────

@dataclass
class CurriculumTask:
    """A single training task assignment produced by the scheduler."""

    task_id: str
    max_steps: int
    noise_profile: NoiseProfile
    n_rows: int
    difficulty: str  # "easy" | "medium" | "hard"
    seed: Optional[int] = None

    def reset_kwargs(self) -> dict[str, Any]:
        """Keyword arguments suitable for passing to env.reset()."""
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "dataset_n_rows": self.n_rows,
            "noise_profile_override": self.noise_profile,
        }


# ── CurriculumScheduler ────────────────────────────────────────────────────────

class CurriculumScheduler:
    """Assigns training tasks and promotes difficulty as the agent improves.

    Parameters
    ----------
    task_ids:
        Task IDs to rotate through. Defaults to all three built-in tasks.
    mode:
        ``"round_robin"`` — cycle tasks in order (default).
        ``"random"``      — uniform random pick each episode.
    start_level:
        Starting difficulty level name: ``"easy"``, ``"medium"``, or ``"hard"``.
    rng:
        Optional ``random.Random`` instance for reproducibility.
    """

    def __init__(
        self,
        task_ids: Optional[List[str]] = None,
        mode: str = "round_robin",
        start_level: str = "easy",
        rng: Optional[random.Random] = None,
    ) -> None:
        self._task_ids: List[str] = task_ids or list(ALL_TASK_IDS)
        if not self._task_ids:
            raise ValueError("task_ids must be non-empty")
        if mode not in ("round_robin", "random"):
            raise ValueError(f"Unknown mode: {mode!r}. Use 'round_robin' or 'random'.")
        self._mode = mode
        self._rng = rng or random.Random()

        self._level_index: int = self._name_to_index(start_level)
        self._task_index: int = 0  # for round_robin

        # Rolling window of success flags for the current difficulty level.
        self._window: Deque[bool] = deque()
        self._episode_count: int = 0
        self._promotions: List[dict[str, Any]] = []

    # ── Public interface ───────────────────────────────────────────────────────

    @property
    def current_level(self) -> DifficultyLevel:
        return DIFFICULTY_LEVELS[self._level_index]

    @property
    def level_name(self) -> str:
        return self.current_level.name

    @property
    def at_max_difficulty(self) -> bool:
        return self._level_index >= len(DIFFICULTY_LEVELS) - 1

    def next_task(self, seed: Optional[int] = None) -> CurriculumTask:
        """Return the next task assignment for the current difficulty stage."""
        level = self.current_level
        task_id = self._pick_task()
        return CurriculumTask(
            task_id=task_id,
            max_steps=level.max_steps,
            noise_profile=level.noise_profile,
            n_rows=level.n_rows,
            difficulty=level.name,
            seed=seed,
        )

    def record_episode(self, success: bool) -> bool:
        """Record episode outcome. Returns True if a promotion just occurred."""
        level = self.current_level
        self._window.append(success)
        if len(self._window) > level.window_size:
            self._window.popleft()
        self._episode_count += 1
        return self._maybe_promote()

    def summary(self) -> dict[str, Any]:
        """Serializable scheduler state for logging."""
        level = self.current_level
        window_success_rate = (
            sum(self._window) / len(self._window) if self._window else 0.0
        )
        return {
            "current_difficulty": level.name,
            "level_index": self._level_index,
            "episode_count": self._episode_count,
            "window_size": len(self._window),
            "window_success_rate": round(window_success_rate, 4),
            "promotion_threshold": level.promotion_threshold,
            "at_max_difficulty": self.at_max_difficulty,
            "promotions": list(self._promotions),
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _name_to_index(name: str) -> int:
        for i, lvl in enumerate(DIFFICULTY_LEVELS):
            if lvl.name == name:
                return i
        raise ValueError(f"Unknown difficulty level: {name!r}. Choose from {[l.name for l in DIFFICULTY_LEVELS]}")

    def _pick_task(self) -> str:
        if self._mode == "random":
            return self._rng.choice(self._task_ids)
        # round_robin
        task_id = self._task_ids[self._task_index % len(self._task_ids)]
        self._task_index += 1
        return task_id

    def _maybe_promote(self) -> bool:
        level = self.current_level
        if self.at_max_difficulty:
            return False
        if len(self._window) < level.window_size:
            return False
        rate = sum(self._window) / len(self._window)
        if rate >= level.promotion_threshold:
            old_name = level.name
            self._level_index += 1
            self._window.clear()
            self._promotions.append(
                {
                    "episode": self._episode_count,
                    "from": old_name,
                    "to": self.current_level.name,
                    "trigger_rate": round(rate, 4),
                }
            )
            return True
        return False


__all__ = [
    "DifficultyLevel",
    "DIFFICULTY_LEVELS",
    "LEVELS_BY_NAME",
    "ALL_TASK_IDS",
    "CurriculumTask",
    "CurriculumScheduler",
]
