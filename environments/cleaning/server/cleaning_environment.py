"""
DataSage Cleaning Environment Implementation.

An RL environment where the agent must clean corrupted enterprise data
across 4 domains (HR, Sales, PM, IT Ops) by applying cleaning operations
to maximise the data quality score.
"""

import random
import sys
import os

from uuid import uuid4

import numpy as np
import pandas as pd

# Allow imports from the project root so shared modules are reachable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from environments.shared.domains import DOMAINS
from environments.shared.enterprise_data import (
    load_domain_data,
    inject_corruption,
    compute_dq_score,
    format_preview,
    format_columns_info,
)
from environments.shared.reward_utils import cleaning_reward

from models import CleaningAction, CleaningObservation
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State


class CleaningEnvironment(Environment):
    """
    Cleaning environment: the agent fixes data quality issues in a
    50-row enterprise data batch.

    Supported operations:
        fill_null   - fill missing values (median / mode / explicit value)
        fix_type    - cast a column to numeric, coercing errors to NaN
        remove_duplicate - drop duplicate rows
        standardize - strip whitespace and normalise case (lower / title)
        trim        - strip leading/trailing whitespace
        correct_typo - replace a typo with a correct value
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialise the cleaning environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._df: pd.DataFrame = pd.DataFrame()
        self._domain_name: str = ""
        self._domain_config = None
        self._max_steps: int = 15

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, domain: str | None = None) -> CleaningObservation:
        """Pick a domain, load a 50-row batch, inject corruption.

        Args:
            seed: If provided, seeds both ``random`` and ``np.random`` before
                  any stochastic operation so the reset is fully reproducible.
            domain: If provided (and valid), use this domain instead of a
                    random choice.
        """
        # Seed RNGs for reproducibility when requested
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Domain selection
        if domain is not None and domain in DOMAINS:
            self._domain_name = domain
        else:
            self._domain_name = random.choice(list(DOMAINS.keys()))
        self._domain_config = DOMAINS[self._domain_name]

        # Load raw data and sample 50 rows.
        # When a seed is provided, temporarily patch np.random.default_rng so
        # that downstream helpers (load_domain_data, inject_corruption) which
        # create their own RNG via default_rng(42) instead receive a generator
        # seeded with *our* seed, making corruption fully reproducible.
        raw_df = load_domain_data(self._domain_name, sample_size=50)
        if seed is not None:
            _orig_default_rng = np.random.default_rng
            np.random.default_rng = lambda *a, **kw: _orig_default_rng(seed)
            try:
                self._df = inject_corruption(raw_df, self._domain_config, rate=0.15)
            finally:
                np.random.default_rng = _orig_default_rng
        else:
            self._df = inject_corruption(raw_df, self._domain_config, rate=0.15)

        dq = compute_dq_score(self._df, self._domain_config)
        dq_report = (
            f"completeness={dq['completeness']:.4f}  "
            f"consistency={dq['consistency']:.4f}  "
            f"uniqueness={dq['uniqueness']:.4f}"
        )

        # Store initial DQ for later improvement-delta rewards
        self._initial_dq = dq["overall"]

        return CleaningObservation(
            domain=self._domain_name,
            data_preview=format_preview(self._df),
            dq_report=dq_report,
            dq_score=dq["overall"],
            columns_info=format_columns_info(self._df, self._domain_config),
            step_number=0,
            max_steps=self._max_steps,
            done=False,
            reward=0.0,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: CleaningAction) -> CleaningObservation:  # type: ignore[override]
        """Apply a single cleaning operation and return the updated state."""
        self._state.step_count += 1
        step = self._state.step_count

        op = action.operation
        col = action.column
        value = action.value

        # ---- apply operation ----
        try:
            if op == "fill_null":
                self._apply_fill_null(col, value)
            elif op == "fix_type":
                self._apply_fix_type(col)
            elif op == "remove_duplicate":
                self._apply_remove_duplicate()
            elif op == "standardize":
                self._apply_standardize(col, value)
            elif op == "trim":
                self._apply_trim(col)
            elif op == "correct_typo":
                self._apply_correct_typo(col, value, action.params)
            # unknown ops are silently ignored (no crash)
        except Exception:
            pass  # invalid column, etc. -> no-op

        # ---- compute DQ and reward ----
        dq = compute_dq_score(self._df, self._domain_config)
        reward = cleaning_reward(self._initial_dq, dq["overall"])
        done = dq["overall"] > 0.95 or step >= self._max_steps

        dq_report = (
            f"completeness={dq['completeness']:.4f}  "
            f"consistency={dq['consistency']:.4f}  "
            f"uniqueness={dq['uniqueness']:.4f}"
        )

        return CleaningObservation(
            domain=self._domain_name,
            data_preview=format_preview(self._df),
            dq_report=dq_report,
            dq_score=dq["overall"],
            columns_info=format_columns_info(self._df, self._domain_config),
            step_number=step,
            max_steps=self._max_steps,
            done=done,
            reward=reward,
            metadata={
                "operation": op,
                "column": col,
                "step": step,
            },
        )

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        """Return current environment state."""
        return self._state

    # ------------------------------------------------------------------
    # operation helpers
    # ------------------------------------------------------------------
    def _apply_fill_null(self, col: str, value: str | None) -> None:
        if col not in self._df.columns:
            return
        if value == "median":
            numeric = pd.to_numeric(self._df[col], errors="coerce")
            fill_val = numeric.median()
        elif value == "mode":
            mode_vals = self._df[col].mode()
            fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else ""
        else:
            fill_val = value if value is not None else ""
        self._df[col] = self._df[col].fillna(fill_val)

    def _apply_fix_type(self, col: str) -> None:
        if col not in self._df.columns:
            return
        self._df[col] = pd.to_numeric(self._df[col], errors="coerce")

    def _apply_remove_duplicate(self) -> None:
        available = [c for c in self._domain_config.columns if c in self._df.columns]
        self._df = self._df.drop_duplicates(
            subset=available[:5], keep="first"
        ).reset_index(drop=True)

    def _apply_standardize(self, col: str, value: str | None) -> None:
        if col not in self._df.columns:
            return
        self._df[col] = self._df[col].astype(str).str.strip()
        if value == "lower":
            self._df[col] = self._df[col].str.lower()
        elif value == "title":
            self._df[col] = self._df[col].str.title()

    def _apply_trim(self, col: str) -> None:
        if col not in self._df.columns:
            return
        self._df[col] = self._df[col].astype(str).str.strip()

    def _apply_correct_typo(self, col: str, value: str | None,
                            params: dict) -> None:
        if col not in self._df.columns or value is None:
            return
        wrong = params.get("wrong")
        if wrong:
            self._df[col] = self._df[col].replace(wrong, value)
        else:
            # If no specific wrong value given, try to replace the most
            # uncommon value with the provided correct value.
            counts = self._df[col].value_counts()
            if len(counts) > 1:
                least_common = counts.index[-1]
                self._df[col] = self._df[col].replace(least_common, value)
