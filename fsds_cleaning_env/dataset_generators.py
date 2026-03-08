"""Synthetic dataset generators for the FSDS Cleaning Environment.

Generates larger, diverse tables with configurable noise to maximize RL learning:
- Per-episode generation (different table each episode) for training
- Fixed seeds for evaluation and debugging
- Realistic pathologies: missingness, invalid tokens, duplicates, outliers, category drift
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


# Invalid tokens that the environment recognizes and expects agents to handle.
INVALID_TOKENS = {"", " ", "unknown", "UNKNOWN", "n/a", "N/A", "null", "NULL", "?", "--"}


@dataclass
class NoiseProfile:
    """Controls the mix of pathologies injected into synthetic data.

    All probabilities are per-cell (or per-row for duplicates), applied
    independently. Values should be in [0, 1].
    """

    p_missing: float = 0.05
    p_invalid_token: float = 0.04
    p_duplicate_row: float = 0.03
    p_outlier: float = 0.02
    p_category_drift: float = 0.06  # whitespace, case, aliases
    p_string_in_numeric: float = 0.03  # store numeric as string with junk

    @classmethod
    def light(cls) -> NoiseProfile:
        """Mild noise for easier episodes."""
        return cls(
            p_missing=0.02,
            p_invalid_token=0.02,
            p_duplicate_row=0.02,
            p_outlier=0.01,
            p_category_drift=0.03,
            p_string_in_numeric=0.02,
        )

    @classmethod
    def medium(cls) -> NoiseProfile:
        """Default training noise."""
        return cls()

    @classmethod
    def heavy(cls) -> NoiseProfile:
        """Challenging noise for harder episodes."""
        return cls(
            p_missing=0.08,
            p_invalid_token=0.06,
            p_duplicate_row=0.05,
            p_outlier=0.04,
            p_category_drift=0.10,
            p_string_in_numeric=0.05,
        )


# Default sizes
SIZE_DEBUG = 12
SIZE_SMALL = 100
SIZE_MEDIUM = 500
SIZE_LARGE = 1000


def _apply_noise(
    df: pd.DataFrame,
    seed: int,
    profile: NoiseProfile,
    numeric_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    skip_missing_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Inject noise into a clean DataFrame. Does not modify the target column.

    Args:
        skip_missing_columns: Columns to exclude from missing-value injection
            (e.g. ID columns and datetime columns that have no impute operation).
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    _skip_missing = set(skip_missing_columns or [])

    # 1. Missing values (exclude target and skip_missing_columns)
    for col in out.columns:
        if col == target_column or col in _skip_missing:
            continue
        mask = rng.random(len(out)) < profile.p_missing
        if mask.any():
            out.loc[mask, col] = np.nan

    # 2. Invalid tokens in object columns
    for col in categorical_columns:
        if col not in out.columns or col == target_column:
            continue
        invalids = list(INVALID_TOKENS)
        mask = rng.random(len(out)) < profile.p_invalid_token
        if mask.any():
            out.loc[mask, col] = rng.choice(invalids, size=mask.sum())

    # 3. Duplicate rows
    if profile.p_duplicate_row > 0:
        n_dup = max(1, int(len(out) * profile.p_duplicate_row))
        dup_idx = rng.choice(out.index, size=min(n_dup, len(out)), replace=True)
        dup_rows = out.loc[dup_idx]
        out = pd.concat([out, dup_rows], ignore_index=True)

    # 4. Outliers in numeric columns
    for col in numeric_columns:
        if col not in out.columns or col == target_column:
            continue
        try:
            vals = pd.to_numeric(out[col], errors="coerce")
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                mask = rng.random(len(out)) < profile.p_outlier
                if mask.any():
                    # Inject extreme values
                    extreme_high = q3 + 10 * iqr
                    extreme_low = q1 - 10 * iqr
                    out.loc[mask, col] = rng.choice([extreme_high, extreme_low], size=mask.sum())
        except Exception:
            pass

    # 5. Category drift (whitespace, case, aliases)
    if profile.p_category_drift > 0:
        for col in categorical_columns:
            if col not in out.columns or col == target_column:
                continue
            mask = rng.random(len(out)) < profile.p_category_drift
            if mask.any():
                # Add whitespace or change case
                s = out[col].astype(str)
                drift = rng.choice(["strip", "upper", "lower", "space"], size=mask.sum())
                for i, idx in enumerate(out.index[mask]):
                    if drift[i] == "strip":
                        out.loc[idx, col] = "  " + str(s.loc[idx]) + "  "
                    elif drift[i] == "upper":
                        out.loc[idx, col] = str(s.loc[idx]).upper()
                    elif drift[i] == "lower":
                        out.loc[idx, col] = str(s.loc[idx]).lower()
                    else:
                        out.loc[idx, col] = str(s.loc[idx]) + " "

    # 6. String-in-numeric: store numbers as strings with occasional junk
    for col in numeric_columns:
        if col not in out.columns or col == target_column:
            continue
        mask = rng.random(len(out)) < profile.p_string_in_numeric
        if mask.any():
            # Convert column to object dtype to allow mixed types, then inject junk
            out[col] = out[col].astype(object)
            junk = rng.choice(["unknown", "?", "n/a", ""], size=mask.sum())
            out.loc[mask, col] = junk

    return out.reset_index(drop=True)


def generate_mobile_ecommerce(
    n_rows: int = SIZE_MEDIUM,
    seed: Optional[int] = None,
    noise_profile: Optional[NoiseProfile] = None,
) -> pd.DataFrame:
    """Generate a mobile conversion dataset with configurable noise."""
    rng = np.random.default_rng(seed)
    profile = noise_profile or NoiseProfile.medium()

    devices = ["ios", "android", "web"]
    countries = ["US", "MX", "CA", "BR", "CO"]
    base_date = datetime(2026, 1, 1)

    sessions = list(range(1, n_rows + 1))
    device_os = rng.choice(devices, size=n_rows)
    customer_id = [str(1000 + i) for i in range(n_rows)]
    country = rng.choice(countries, size=n_rows)
    items_in_cart = rng.integers(0, 10, size=n_rows).astype(float)
    order_value = (rng.exponential(30, size=n_rows) + 10).astype(float)
    event_date = [base_date + timedelta(days=int(d)) for d in rng.integers(0, 90, size=n_rows)]
    converted = rng.integers(0, 2, size=n_rows)

    df = pd.DataFrame({
        "session_id": sessions,
        "device_os": device_os,
        "customer_id": customer_id,
        "country": country,
        "items_in_cart": items_in_cart,
        "order_value": order_value,
        "event_date": [d.strftime("%Y-%m-%d") for d in event_date],
        "converted": converted,
    })

    return _apply_noise(
        df,
        seed=seed or rng.integers(0, 2**31),
        profile=profile,
        numeric_columns=["items_in_cart", "order_value"],
        categorical_columns=["device_os", "country"],
        target_column="converted",
        skip_missing_columns=["session_id", "customer_id", "event_date"],
    )


def generate_subscription_churn(
    n_rows: int = SIZE_MEDIUM,
    seed: Optional[int] = None,
    noise_profile: Optional[NoiseProfile] = None,
) -> pd.DataFrame:
    """Generate a subscription churn dataset with configurable noise."""
    rng = np.random.default_rng(seed)
    profile = noise_profile or NoiseProfile.medium()

    plan_types = ["monthly", "annual"]
    payment_methods = ["credit_card", "paypal", "bank_transfer", "debit"]

    customer_key = [f"C{i:04d}" for i in range(n_rows)]
    age = (rng.integers(18, 70, size=n_rows)).astype(float)
    monthly_spend = (rng.exponential(200, size=n_rows) + 50).astype(float)
    plan_type = rng.choice(plan_types, size=n_rows)
    tenure_months = rng.integers(0, 36, size=n_rows).astype(float)
    payment_method = rng.choice(payment_methods, size=n_rows)
    churned = rng.integers(0, 2, size=n_rows)

    df = pd.DataFrame({
        "customer_key": customer_key,
        "age": age,
        "monthly_spend": monthly_spend,
        "plan_type": plan_type,
        "tenure_months": tenure_months,
        "payment_method": payment_method,
        "churned": churned,
    })

    return _apply_noise(
        df,
        seed=seed or rng.integers(0, 2**31),
        profile=profile,
        numeric_columns=["age", "monthly_spend", "tenure_months"],
        categorical_columns=["plan_type", "payment_method"],
        target_column="churned",
        skip_missing_columns=["customer_key"],
    )


def generate_delivery_eta(
    n_rows: int = SIZE_MEDIUM,
    seed: Optional[int] = None,
    noise_profile: Optional[NoiseProfile] = None,
) -> pd.DataFrame:
    """Generate a last-mile delivery ETA dataset with configurable noise."""
    rng = np.random.default_rng(seed)
    profile = noise_profile or NoiseProfile.medium()

    cities = ["Monterrey", "CDMX", "GDL", "MTY"]
    vehicles = ["bike", "car", "motorbike"]

    route_id = [f"R{i:03d}" for i in range(n_rows)]
    city = rng.choice(cities, size=n_rows)
    driver_rating = (rng.uniform(3.0, 5.0, size=n_rows)).astype(float)
    delivery_distance_km = (rng.uniform(5, 40, size=n_rows)).astype(float)
    late_deliveries_last_30d = rng.integers(0, 5, size=n_rows).astype(float)
    vehicle_type = rng.choice(vehicles, size=n_rows)
    delivery_time_minutes = (
        15 + delivery_distance_km * 0.8 + rng.normal(0, 3, size=n_rows)
    ).clip(10, 60).astype(float)

    df = pd.DataFrame({
        "route_id": route_id,
        "city": city,
        "driver_rating": driver_rating,
        "delivery_distance_km": delivery_distance_km,
        "late_deliveries_last_30d": late_deliveries_last_30d,
        "vehicle_type": vehicle_type,
        "delivery_time_minutes": delivery_time_minutes,
    })

    return _apply_noise(
        df,
        seed=seed or rng.integers(0, 2**31),
        profile=profile,
        numeric_columns=["driver_rating", "delivery_distance_km", "late_deliveries_last_30d"],
        categorical_columns=["city", "vehicle_type"],
        target_column="delivery_time_minutes",
        skip_missing_columns=["route_id"],
    )


# Map task_id to generator function
GENERATORS: dict[str, Callable[..., pd.DataFrame]] = {
    "ecommerce_mobile": generate_mobile_ecommerce,
    "subscription_churn": generate_subscription_churn,
    "delivery_eta": generate_delivery_eta,
}


def _static_mobile_ecommerce() -> pd.DataFrame:
    """Original tiny static dataset for debugging."""
    rows = [
        [1, "ios", "1001", "US", "1", 49.99, "2026-01-01", 1],
        [2, "android", "1002", "MX", "2", 19.99, "2026-01-02", 0],
        [3, "ios", "1003", "unknown", "1", 39.99, "2026-01-03", 1],
        [4, "android", "1004", "MX", "", 15.00, "2026-01-03", 0],
        [4, "android", "1004", "MX", "", 15.00, "2026-01-03", 0],
        [5, "web", "1005", "US", "999", 10.00, "2026-01-04", 0],
        [6, "ios", "1006", "CA", "1", 5000.0, "2026-01-04", 1],
        [7, "Android ", "1007", "ca", "3", 29.00, "2026-01-05", 1],
        [8, "iOS", "1008", "MX", "2", 35.00, "2026-01-05", 1],
        [9, "web", "1009", "?", "1", 22.00, "2026-01-06", 0],
        [10, "ios", "1010", "US", None, 18.50, "2026-01-07", 0],
        [11, "android", "1011", "MX", "2", 25.00, "2026-01-08", 1],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "session_id", "device_os", "customer_id", "country",
            "items_in_cart", "order_value", "event_date", "converted",
        ],
    )


def _static_subscription() -> pd.DataFrame:
    """Original tiny static dataset for debugging."""
    rows = [
        ["C001", "29", "299", "monthly", "2", "credit_card", 0],
        ["C002", "31", "unknown", "monthly", "6", "paypal", 1],
        ["C003", " ", "499", "annual", "12", "credit_card", 0],
        ["C004", "45", "199", "Monthly", "1", "bank_transfer", 1],
        ["C004", "45", "199", "Monthly", "1", "bank_transfer", 1],
        ["C005", "27", "149", "monthly", "0", "paypal", 0],
        ["C006", "52", "20000", "annual", "24", "credit_card", 1],
        ["C007", None, "259", "annual", "3", "credit_card", 0],
        ["C008", "39", "--", "monthly", "?", "debit", 1],
        ["C009", "41", "399", "annual", "10", "credit_card", 0],
        ["C010", "38", "350", "monthly", "5", "paypal", 0],
        ["C011", "36", "499", "annual", "7", "credit_card", 1],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "customer_key", "age", "monthly_spend", "plan_type",
            "tenure_months", "payment_method", "churned",
        ],
    )


def _static_delivery() -> pd.DataFrame:
    """Original tiny static dataset for debugging."""
    rows = [
        ["R001", "Monterrey", "4.2", "12", "2", "bike", 19.0],
        ["R002", "CDMX", "3.8", "25", "3", "car", 31.0],
        ["R003", "GDL", "", "18", "2", "bike", 27.0],
        ["R004", "Monterrey ", "4.9", "999", "2", "bike", 20.5],
        ["R005", "MTY", "4.1", "11", "1", "motorbike", 16.5],
        ["R006", "CDMX", "unknown", "21", "4", "car", 35.0],
        ["R007", "GDL", "3.5", "19", "2", "bike", 29.0],
        ["R008", "GDL", "3.5", "19", "2", "bike", 29.0],
        ["R009", "Monterrey", None, "10", "2", "bike", 18.0],
        ["R010", "CDMX", "4.7", "16", "?", "car", 26.0],
        ["R011", "Monterrey", "4.0", "14", "1", "bike", 22.0],
        ["R012", "CDMX", "4.2", "17", "2", "CAR", 24.5],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "route_id", "city", "driver_rating", "delivery_distance_km",
            "late_deliveries_last_30d", "vehicle_type", "delivery_time_minutes",
        ],
    )


STATIC_DATASETS: dict[str, Callable[[], pd.DataFrame]] = {
    "ecommerce_mobile": _static_mobile_ecommerce,
    "subscription_churn": _static_subscription,
    "delivery_eta": _static_delivery,
}


def make_dataset_factory(
    task_id: str,
    n_rows: int = SIZE_MEDIUM,
    noise_profile: Optional[NoiseProfile] = None,
    use_static_fallback: bool = False,
) -> Callable[..., pd.DataFrame]:
    """Create a dataset factory callable for use in TaskSpec.

    The returned callable accepts:
        seed: Optional[int] — for reproducibility. If None, each call produces different data.
        n_rows_override: Optional[int] — override default row count.
        noise_profile_override: Optional[NoiseProfile] — override default noise.

    Returns a DataFrame. For training, pass seed=None to get a fresh table each episode.
    For evaluation, pass a fixed seed to get reproducible held-out data.
    """

    generator = GENERATORS.get(task_id)
    if generator is None:
        raise ValueError(f"Unknown task_id: {task_id}")

    def factory(
        seed: Optional[int] = None,
        n_rows_override: Optional[int] = None,
        noise_profile_override: Optional[NoiseProfile] = None,
        dataset_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if dataset_mode == "debug" or (use_static_fallback and n_rows <= SIZE_DEBUG):
            return STATIC_DATASETS[task_id]().copy()

        n = n_rows_override or n_rows
        prof = noise_profile_override or noise_profile or NoiseProfile.medium()
        return generator(n_rows=n, seed=seed, noise_profile=prof)

    return factory


# Fixed seeds for evaluation. Each eval task gets a deterministic table.
EVAL_SEEDS: dict[str, list[int]] = {
    "ecommerce_mobile": [42, 101, 202, 303, 404],
    "subscription_churn": [43, 102, 203, 304, 405],
    "delivery_eta": [44, 103, 204, 305, 406],
}


def get_eval_dataset(task_id: str, eval_index: int = 0, n_rows: int = SIZE_MEDIUM) -> pd.DataFrame:
    """Get a fixed held-out dataset for evaluation.

    Uses EVAL_SEEDS so the same table is produced every time for a given task and index.
    """
    seeds = EVAL_SEEDS.get(task_id, [42])
    seed = seeds[eval_index % len(seeds)]
    gen = GENERATORS[task_id]
    return gen(n_rows=n_rows, seed=seed, noise_profile=NoiseProfile.medium())


__all__ = [
    "NoiseProfile",
    "SIZE_DEBUG",
    "SIZE_SMALL",
    "SIZE_MEDIUM",
    "SIZE_LARGE",
    "generate_mobile_ecommerce",
    "generate_subscription_churn",
    "generate_delivery_eta",
    "GENERATORS",
    "make_dataset_factory",
    "EVAL_SEEDS",
    "get_eval_dataset",
]
