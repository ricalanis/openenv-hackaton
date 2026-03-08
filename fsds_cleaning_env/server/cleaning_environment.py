"""FSDS Cleaning Environment.

This is an MCP/OpenEnv environment designed for the Silver-layer of the user's
Full-Stack Data Science thesis: a DS Agent cleans and validates messy tabular
business data while a QA/PCS layer enforces deterministic quality gates.

The environment exposes tools rather than a fixed action schema so that LLM
agents can inspect the dataset, choose cleaning operations, run gates, and
submit a final cleaned table.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from fastmcp import FastMCP
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from fsds_cleaning_env.dataset_generators import (
    SIZE_MEDIUM,
    make_dataset_factory,
)
from fsds_cleaning_env.reward import (
    FinalRewardInput,
    StepRewardInput,
    TOOL_ERROR_REWARD,
    compute_final_reward,
    compute_quality_gate_bonus,
    compute_step_reward,
)


try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State


INVALID_TOKENS = {"", " ", "unknown", "UNKNOWN", "n/a", "N/A", "null", "NULL", "?", "--"}

# Observation metadata schema version for downstream agents.
OBSERVATION_SCHEMA_VERSION = "1.0"

# Allowed operations for the apply_cleaning_operation tool. This defines the
# logical action space that agents should use.
AVAILABLE_OPERATIONS = [
    "drop_duplicates",
    "replace_invalid_with_null",
    "cast_numeric",
    "cast_datetime",
    "impute_numeric",
    "impute_categorical",
    "normalize_categories",
    "clip_outliers_iqr",
]


@dataclass
class TaskSpec:
    task_id: str
    title: str
    objective: str
    business_context: str
    target_column: str
    task_type: str  # classification | regression
    dataset_factory: Any
    expected_types: dict[str, str]
    required_ops: list[dict[str, Any]]
    allowed_columns_to_drop: list[str] = field(default_factory=list)
    min_retention_ratio: float = 0.85
    notes: list[str] = field(default_factory=list)


@dataclass
class EpisodeData:
    spec: TaskSpec
    raw_df: pd.DataFrame
    working_df: pd.DataFrame
    operation_log: list[dict[str, Any]] = field(default_factory=list)
    last_gate_report: dict[str, Any] = field(default_factory=dict)
    total_reward: float = 0.0
    submitted: bool = False
    max_steps: int = 18


TASKS: dict[str, TaskSpec] = {
    "ecommerce_mobile": TaskSpec(
        task_id="ecommerce_mobile",
        title="Mobile conversion cleaning",
        objective="Prepare a mobile conversion table for downstream churn/conversion modeling.",
        business_context=(
            "Marketing needs a trustworthy Silver table to investigate why mobile conversions fell. "
            "Clean the dataset without destroying row retention."
        ),
        target_column="converted",
        task_type="classification",
        dataset_factory=make_dataset_factory("ecommerce_mobile", n_rows=SIZE_MEDIUM),
        expected_types={
            "session_id": "int64",
            "device_os": "str",
            "customer_id": "str",
            "country": "str",
            "items_in_cart": "float64",
            "order_value": "float64",
            "event_date": "datetime64[us]",
            "converted": "int64",
        },
        required_ops=[
            {"operation": "replace_invalid_with_null", "column": "country"},
            {"operation": "replace_invalid_with_null", "column": "items_in_cart"},
            {"operation": "replace_invalid_with_null", "column": "device_os"},
            {"operation": "cast_numeric", "column": "items_in_cart"},
            {"operation": "cast_numeric", "column": "order_value"},
            {"operation": "impute_numeric", "column": "items_in_cart"},
            {"operation": "impute_numeric", "column": "order_value"},
            {"operation": "clip_outliers_iqr", "column": "items_in_cart"},
            {"operation": "clip_outliers_iqr", "column": "order_value"},
            {"operation": "normalize_categories", "column": "device_os"},
            {"operation": "normalize_categories", "column": "country"},
            {"operation": "impute_categorical", "column": "device_os"},
            {"operation": "impute_categorical", "column": "country"},
            {"operation": "cast_datetime", "column": "event_date"},
            {"operation": "drop_duplicates"},
        ],
        notes=[
            "Preserve the target column.",
            "Do not drop high-value rows unless a quality gate demands it.",
        ],
    ),
    "subscription_churn": TaskSpec(
        task_id="subscription_churn",
        title="Subscription churn cleaning",
        objective="Create a clean churn-ready subscriber table.",
        business_context=(
            "Retention wants to model churn risk, but the raw CRM extract has duplicated customers, "
            "invalid tokens, and numeric columns stored as strings."
        ),
        target_column="churned",
        task_type="classification",
        dataset_factory=make_dataset_factory("subscription_churn", n_rows=SIZE_MEDIUM),
        expected_types={
            "customer_key": "str",
            "age": "float64",
            "monthly_spend": "float64",
            "plan_type": "str",
            "tenure_months": "float64",
            "payment_method": "str",
            "churned": "int64",
        },
        required_ops=[
            {"operation": "replace_invalid_with_null", "column": "monthly_spend"},
            {"operation": "replace_invalid_with_null", "column": "age"},
            {"operation": "replace_invalid_with_null", "column": "tenure_months"},
            {"operation": "replace_invalid_with_null", "column": "payment_method"},
            {"operation": "cast_numeric", "column": "age"},
            {"operation": "cast_numeric", "column": "monthly_spend"},
            {"operation": "cast_numeric", "column": "tenure_months"},
            {"operation": "impute_numeric", "column": "age"},
            {"operation": "impute_numeric", "column": "monthly_spend"},
            {"operation": "impute_numeric", "column": "tenure_months"},
            {"operation": "clip_outliers_iqr", "column": "monthly_spend"},
            {"operation": "normalize_categories", "column": "plan_type"},
            {"operation": "normalize_categories", "column": "payment_method"},
            {"operation": "impute_categorical", "column": "plan_type"},
            {"operation": "impute_categorical", "column": "payment_method"},
            {"operation": "drop_duplicates"},
        ],
        notes=["Monthly spend contains a severe outlier that should be handled, not ignored."],
    ),
    "delivery_eta": TaskSpec(
        task_id="delivery_eta",
        title="Delivery ETA cleaning",
        objective="Clean a last-mile delivery table before ETA modeling.",
        business_context=(
            "Operations needs a reliable table to predict ETA. The raw export mixes city aliases, "
            "rating gaps, duplicated rows and suspicious distance values."
        ),
        target_column="delivery_time_minutes",
        task_type="regression",
        dataset_factory=make_dataset_factory("delivery_eta", n_rows=SIZE_MEDIUM),
        expected_types={
            "route_id": "str",
            "city": "str",
            "driver_rating": "float64",
            "delivery_distance_km": "float64",
            "late_deliveries_last_30d": "float64",
            "vehicle_type": "str",
            "delivery_time_minutes": "float64",
        },
        required_ops=[
            {"operation": "replace_invalid_with_null", "column": "driver_rating"},
            {"operation": "replace_invalid_with_null", "column": "late_deliveries_last_30d"},
            {"operation": "replace_invalid_with_null", "column": "city"},
            {"operation": "replace_invalid_with_null", "column": "vehicle_type"},
            {"operation": "cast_numeric", "column": "driver_rating"},
            {"operation": "cast_numeric", "column": "delivery_distance_km"},
            {"operation": "cast_numeric", "column": "late_deliveries_last_30d"},
            {"operation": "impute_numeric", "column": "driver_rating"},
            {"operation": "impute_numeric", "column": "late_deliveries_last_30d"},
            {"operation": "impute_numeric", "column": "delivery_distance_km"},
            {"operation": "clip_outliers_iqr", "column": "delivery_distance_km"},
            {"operation": "normalize_categories", "column": "city"},
            {"operation": "normalize_categories", "column": "vehicle_type"},
            {"operation": "impute_categorical", "column": "city"},
            {"operation": "impute_categorical", "column": "vehicle_type"},
            {"operation": "drop_duplicates"},
        ],
        notes=["City aliases should be standardized before downstream feature engineering."],
    ),
}


class FSDSCleaningEnvironment(MCPEnvironment):
    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode: Optional[EpisodeData] = None

        mcp = FastMCP("fsds_cleaning_env")

        @mcp.tool
        def list_tasks() -> dict[str, Any]:
            return {
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "title": task.title,
                        "objective": task.objective,
                        "task_type": task.task_type,
                    }
                    for task in TASKS.values()
                ]
            }

        @mcp.tool
        def get_task_brief() -> dict[str, Any]:
            episode = self._require_episode()
            spec = episode.spec
            return {
                "task_id": spec.task_id,
                "title": spec.title,
                "objective": spec.objective,
                "business_context": spec.business_context,
                "target_column": spec.target_column,
                "success_criteria": {
                    "quality_gate": "pass",
                    "minimum_row_retention": spec.min_retention_ratio,
                    "no_missing_values_outside_allowed": True,
                    "target_preserved": True,
                },
                "notes": spec.notes,
            }

        @mcp.tool
        def preview_data(n: int = 5) -> dict[str, Any]:
            episode = self._require_episode()
            return {
                "rows": episode.working_df.head(n).to_dict(orient="records"),
                "shape": list(episode.working_df.shape),
            }

        @mcp.tool
        def profile_data() -> dict[str, Any]:
            episode = self._require_episode()
            df = episode.working_df
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_summary = {}
            if numeric_cols:
                desc = df[numeric_cols].describe(include="all").fillna("NaN")
                numeric_summary = desc.to_dict()
            return {
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "missing_counts": df.isna().sum().to_dict(),
                "duplicate_rows": int(df.duplicated().sum()),
                "invalid_token_counts": {c: int(df[c].astype(str).isin(INVALID_TOKENS).sum()) for c in df.columns},
                "numeric_summary": numeric_summary,
            }

        @mcp.tool
        def get_operation_history() -> dict[str, Any]:
            episode = self._require_episode()
            return {"operations": episode.operation_log, "total_reward": round(episode.total_reward, 4)}

        @mcp.tool
        def render_episode(n_preview_rows: int = 5) -> dict[str, Any]:
            """Human-friendly snapshot of the current episode state."""
            episode = self._require_episode()
            df = episode.working_df
            return {
                "task_id": episode.spec.task_id,
                "title": episode.spec.title,
                "step_count": self._state.step_count,
                "max_steps": episode.max_steps,
                "total_reward": round(episode.total_reward, 4),
                "shape": list(df.shape),
                "last_gate_report": episode.last_gate_report or None,
                "operations": episode.operation_log,
                "preview": df.head(n_preview_rows).to_dict(orient="records"),
            }

        @mcp.tool
        def apply_cleaning_operation(operation: str, column: Optional[str] = None, strategy: str = "median") -> dict[str, Any]:
            episode = self._require_episode()
            if episode.submitted:
                return self._tool_error("Episode already submitted.")
            if self._state.step_count >= episode.max_steps:
                return self._tool_error("Maximum steps reached. Submit or reset the episode.")

            before_score = self._quality_score(episode)
            before_df = episode.working_df.copy(deep=True)
            message = self._apply_operation(episode, operation=operation, column=column, strategy=strategy)
            after_score = self._quality_score(episode)
            delta = round(after_score - before_score, 4)
            step_reward = compute_step_reward(
                StepRewardInput(quality_before=before_score, quality_after=after_score)
            )
            episode.total_reward += step_reward
            episode.operation_log.append(
                {
                    "operation": operation,
                    "column": column,
                    "strategy": strategy,
                    "message": message,
                    "quality_delta": delta,
                    "reward": round(step_reward, 4),
                    "shape_before": list(before_df.shape),
                    "shape_after": list(episode.working_df.shape),
                }
            )
            return {
                "message": message,
                "reward": round(float(step_reward), 4),
                "quality_score": round(float(after_score), 4),
                "shape": [int(x) for x in episode.working_df.shape],
            }

        @mcp.tool
        def run_quality_gates() -> dict[str, Any]:
            episode = self._require_episode()
            report = self._evaluate_quality_gates(episode)
            episode.last_gate_report = report
            bonus = compute_quality_gate_bonus(bool(report["passed"]))
            episode.total_reward += bonus
            return {
                **report,
                "reward": bonus,
                "total_reward": round(episode.total_reward, 4),
            }

        @mcp.tool
        def submit_solution() -> dict[str, Any]:
            import json as _json
            episode = self._require_episode()
            report = self._evaluate_quality_gates(episode)
            match_score = self._required_operations_score(episode)
            final_reward = round(
                compute_final_reward(
                    FinalRewardInput(
                        quality_score=self._quality_score(episode),
                        gate_passed=bool(report["passed"]),
                        required_operation_coverage=match_score,
                    )
                ),
                4,
            )
            episode.total_reward += final_reward
            episode.submitted = True
            # Use pandas to_json to safely serialize numpy/pandas types (Timestamps, int64, float64).
            cleaned_preview = _json.loads(
                episode.working_df.head(5).to_json(orient="records", date_format="iso")
            )
            return {
                "done": True,
                "passed": bool(report["passed"]),
                "final_reward": final_reward,
                "cumulative_reward": round(float(episode.total_reward), 4),
                "quality_report": report,
                "required_operation_coverage": round(float(match_score), 4),
                "cleaned_preview": cleaned_preview,
            }

        super().__init__(mcp)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        spec = TASKS[task_id] if task_id in TASKS else TASKS["ecommerce_mobile"]
        dataset_mode = kwargs.get("dataset_mode")
        dataset_n_rows = kwargs.get("dataset_n_rows")
        try:
            raw_df = spec.dataset_factory(
                seed=seed,
                dataset_mode=dataset_mode,
                n_rows_override=dataset_n_rows,
            )
        except TypeError:
            raw_df = spec.dataset_factory()
        episode = EpisodeData(spec=spec, raw_df=raw_df.copy(deep=True), working_df=raw_df.copy(deep=True))
        self._episode = episode
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "schema_version": OBSERVATION_SCHEMA_VERSION,
                "task_id": spec.task_id,
                "task_type": spec.task_type,
                "target_column": spec.target_column,
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "max_steps": episode.max_steps,
                "message": f"FSDS cleaning environment ready for task '{spec.task_id}'.",
                "available_tools": [
                    "list_tasks",
                    "get_task_brief",
                    "preview_data",
                    "profile_data",
                    "get_operation_history",
                    "apply_cleaning_operation",
                    "run_quality_gates",
                    "submit_solution",
                    "render_episode",
                ],
                "available_operations": AVAILABLE_OPERATIONS,
            },
        )

    def _step_impl(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        del timeout_s, kwargs
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. Use MCP tools instead."
            },
        )

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state

    def _require_episode(self) -> EpisodeData:
        if self._episode is None:
            raise RuntimeError("Reset the environment before using tools.")
        return self._episode

    def _tool_error(self, message: str) -> dict[str, Any]:
        return {"error": message, "reward": TOOL_ERROR_REWARD}

    def _apply_operation(self, episode: EpisodeData, operation: str, column: Optional[str], strategy: str) -> str:
        df = episode.working_df
        if operation == "drop_duplicates":
            before = len(df)
            episode.working_df = df.drop_duplicates().reset_index(drop=True)
            return f"Dropped {before - len(episode.working_df)} duplicate rows."

        if column is None:
            raise ValueError(f"Operation '{operation}' requires a column.")
        if column not in df.columns:
            raise ValueError(f"Unknown column '{column}'.")

        if operation == "replace_invalid_with_null":
            mask = df[column].astype(str).isin(INVALID_TOKENS)
            changed = int(mask.sum())
            episode.working_df.loc[mask, column] = np.nan
            return f"Replaced {changed} invalid tokens in '{column}' with nulls."

        if operation == "cast_numeric":
            episode.working_df[column] = pd.to_numeric(df[column], errors="coerce")
            return f"Cast '{column}' to numeric with coercion."

        if operation == "cast_datetime":
            episode.working_df[column] = pd.to_datetime(df[column], errors="coerce")
            return f"Cast '{column}' to datetime with coercion."

        if operation == "impute_numeric":
            series = pd.to_numeric(episode.working_df[column], errors="coerce")
            if strategy == "mean":
                fill_value = float(series.mean())
            else:
                fill_value = float(series.median())
            episode.working_df[column] = series.fillna(fill_value)
            return f"Imputed '{column}' using {strategy}={round(fill_value, 4)}."

        if operation == "impute_categorical":
            mode = episode.working_df[column].dropna().astype(str).mode()
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            episode.working_df[column] = episode.working_df[column].fillna(fill_value)
            return f"Imputed '{column}' with mode='{fill_value}'."

        if operation == "normalize_categories":
            null_mask = episode.working_df[column].isna()
            normalized = (
                episode.working_df[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"ios": "ios", "android": "android", "android ": "android", "mty": "monterrey", "car": "car", "CAR": "car"})
            )
            normalized = normalized.replace({
                "ca": "ca",
                "mx": "mx",
                "us": "us",
                "monterrey": "monterrey",
                "cdmx": "cdmx",
                "gdl": "gdl",
                "motorbike": "motorbike",
                "bike": "bike",
            })
            normalized[null_mask] = np.nan
            episode.working_df[column] = normalized
            return f"Normalized categories in '{column}'."

        if operation == "clip_outliers_iqr":
            series = pd.to_numeric(episode.working_df[column], errors="coerce")
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            clipped = series.clip(lower=lower, upper=upper)
            changed = int((series.fillna(clipped) != clipped).sum())
            episode.working_df[column] = clipped
            return f"Clipped {changed} outliers in '{column}' using IQR bounds [{round(lower, 4)}, {round(upper, 4)}]."

        raise ValueError(f"Unsupported operation '{operation}'.")

    def _evaluate_quality_gates(self, episode: EpisodeData) -> dict[str, Any]:
        df = episode.working_df
        raw_df = episode.raw_df
        spec = episode.spec
        missing_counts = df.isna().sum().to_dict()
        non_target_missing = {
            col: int(count)
            for col, count in missing_counts.items()
            if col != spec.target_column and count > 0
        }
        duplicate_rows = int(df.duplicated().sum())
        retention_ratio = len(df) / max(len(raw_df), 1)
        schema_same_columns = list(df.columns) == list(raw_df.columns)
        dtype_alignment = {
            col: str(df[col].dtype) == expected for col, expected in spec.expected_types.items() if col in df.columns
        }
        target_preserved = spec.target_column in df.columns
        stability = self._stability_probe(df, spec)
        tests = {
            "test_missing_values": {"passed": len(non_target_missing) == 0, "details": non_target_missing},
            "test_duplicated_rows": {"passed": duplicate_rows == 0, "details": duplicate_rows},
            "test_data_consistency": {"passed": schema_same_columns and target_preserved, "details": {"schema_same_columns": schema_same_columns, "target_preserved": target_preserved}},
            "test_data_retention": {"passed": retention_ratio >= spec.min_retention_ratio, "details": round(retention_ratio, 4)},
            "test_dtype_alignment": {"passed": all(dtype_alignment.values()), "details": dtype_alignment},
            "test_stability_probe": {"passed": stability["score_std"] <= 0.15, "details": stability},
        }
        passed = all(test["passed"] for test in tests.values())
        return {
            "passed": bool(passed),
            "tests": tests,
            "shape": [int(x) for x in df.shape],
            "retention_ratio": round(float(retention_ratio), 4),
        }

    def _quality_score(self, episode: EpisodeData) -> float:
        df = episode.working_df
        raw_df = episode.raw_df
        spec = episode.spec
        retention = len(df) / max(len(raw_df), 1)
        missing_penalty = df.drop(columns=[spec.target_column], errors="ignore").isna().sum().sum()
        duplicate_penalty = int(df.duplicated().sum())
        dtype_score = 0.0
        for col, expected in spec.expected_types.items():
            if col in df.columns and str(df[col].dtype) == expected:
                dtype_score += 1.0
        dtype_score /= max(len(spec.expected_types), 1)
        score = 1.2 * retention + 0.7 * dtype_score - 0.08 * missing_penalty - 0.2 * duplicate_penalty
        return max(0.0, min(2.0, float(score)))

    def _required_operations_score(self, episode: EpisodeData) -> float:
        executed = [
            {k: v for k, v in op.items() if k in {"operation", "column"} and v is not None}
            for op in episode.operation_log
        ]
        matched = 0
        for required in episode.spec.required_ops:
            req = {k: v for k, v in required.items() if k in {"operation", "column"}}
            if req in executed:
                matched += 1
        return matched / max(len(episode.spec.required_ops), 1)

    def _stability_probe(self, df: pd.DataFrame, spec: TaskSpec) -> dict[str, Any]:
        if spec.target_column not in df.columns or len(df) < 8:
            return {"score_mean": 0.0, "score_std": 1.0, "n_folds": 0}
        clean_df = df.copy()
        X = clean_df.drop(columns=[spec.target_column])
        y = clean_df[spec.target_column]
        X_enc = self._encode_features(X)
        if X_enc.empty:
            return {"score_mean": 0.0, "score_std": 1.0, "n_folds": 0}
        kf = KFold(n_splits=min(3, len(clean_df)), shuffle=True, random_state=42)
        scores: list[float] = []
        for train_idx, test_idx in kf.split(X_enc):
            X_train, X_test = X_enc.iloc[train_idx], X_enc.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            try:
                if spec.task_type == "classification":
                    model = RandomForestClassifier(n_estimators=25, random_state=42)
                    model.fit(X_train, y_train)
                    scores.append(float(model.score(X_test, y_test)))
                else:
                    model = RandomForestRegressor(n_estimators=25, random_state=42)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    mae = np.mean(np.abs(pred - y_test))
                    denom = max(float(np.mean(np.abs(y_test))), 1e-6)
                    scores.append(float(max(0.0, 1.0 - (mae / denom))))
            except Exception:
                scores.append(0.0)
        return {
            "score_mean": round(float(np.mean(scores)), 4) if scores else 0.0,
            "score_std": round(float(np.std(scores)), 4) if scores else 1.0,
            "n_folds": len(scores),
        }

    def _encode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                out[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
            elif pd.api.types.is_datetime64_any_dtype(X[col]):
                out[col] = pd.to_datetime(X[col], errors="coerce").map(lambda x: x.toordinal() if not pd.isna(x) else 0)
            else:
                le = LabelEncoder()
                out[col] = le.fit_transform(X[col].astype(str).fillna("missing"))
        return out
