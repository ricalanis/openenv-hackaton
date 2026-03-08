"""Multi-domain dataset loading, corruption injection, and DQ scoring."""

import random
import string
from typing import Optional

import numpy as np
import pandas as pd

from .domains import DOMAINS, DomainConfig


def load_domain_data(domain: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load domain data from HF dataset or generate synthetic fallback."""
    try:
        from datasets import load_dataset
        ds = load_dataset("ricalanis/datasage-enterprise-raw", domain, split="train")
        df = ds.to_pandas()
    except Exception:
        df = _generate_synthetic(domain)

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df


def _generate_synthetic(domain: str, n: int = 200) -> pd.DataFrame:
    """Generate synthetic data as fallback when HF dataset unavailable."""
    config = DOMAINS[domain]
    rng = np.random.default_rng(42)
    data = {}

    for col in config.columns:
        if col in config.numeric_columns:
            data[col] = rng.normal(50, 20, n).round(2)
        elif col in config.categorical_columns:
            categories = _get_categories(domain, col)
            data[col] = rng.choice(categories, n).tolist()
        elif "ID" in col:
            data[col] = [f"{col[:3].upper()}-{i:04d}" for i in range(n)]
        elif "Date" in col:
            base = pd.Timestamp("2024-01-01")
            data[col] = [(base + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                         for d in rng.integers(0, 365, n)]
        elif "Name" in col or "Assignee" in col or "Rep" in col:
            names = ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Hank"]
            data[col] = rng.choice(names, n).tolist()
        else:
            data[col] = [f"{col}_val_{i}" for i in range(n)]

    return pd.DataFrame(data)


def _get_categories(domain: str, col: str) -> list[str]:
    """Return realistic category values per domain and column."""
    cat_map = {
        "hr": {
            "Department": ["Sales", "Research & Development", "Human Resources"],
            "JobRole": ["Sales Executive", "Research Scientist", "Manager", "Lab Technician",
                        "Manufacturing Director", "Healthcare Representative"],
            "Attrition": ["Yes", "No"],
            "OverTime": ["Yes", "No"],
        },
        "sales": {
            "Stage": ["Prospecting", "Qualification", "Proposal", "Negotiation", "Won", "Lost"],
            "Region": ["East", "West", "Central", "North", "South"],
            "Product": ["GTX Pro", "GTX Basic", "GTX Plus", "MG Special", "MG Advanced"],
            "ForecastCategory": ["Pipeline", "Best Case", "Commit", "Closed"],
        },
        "pm": {
            "Status": ["Not Started", "In Progress", "Completed", "On Hold", "Cancelled"],
            "Priority": ["Critical", "High", "Medium", "Low"],
            "RiskFlag": ["High", "Medium", "Low", "None"],
        },
        "it_ops": {
            "Category": ["Hardware", "Software", "Network", "Access", "Email"],
            "Priority": ["P1-Critical", "P2-High", "P3-Medium", "P4-Low"],
            "Status": ["Open", "In Progress", "Resolved", "Closed", "Pending"],
            "ResolutionType": ["Fix Applied", "Workaround", "No Fix", "Duplicate", "User Error"],
        },
    }
    return cat_map.get(domain, {}).get(col, ["A", "B", "C"])


def inject_corruption(df: pd.DataFrame, domain_config: DomainConfig,
                      rate: float = 0.15) -> pd.DataFrame:
    """Inject realistic data quality issues into a DataFrame."""
    corrupted = df.copy()
    n_rows = len(corrupted)
    rng = np.random.default_rng(42)

    # 1. Inject nulls into numeric columns
    for col in domain_config.numeric_columns:
        if col in corrupted.columns:
            null_mask = rng.random(n_rows) < rate
            corrupted.loc[null_mask, col] = np.nan

    # 2. Inject type mismatches (strings in numeric columns)
    for col in domain_config.numeric_columns:
        if col in corrupted.columns:
            n_bad = max(1, int(n_rows * rate * 0.3))
            bad_idx = rng.choice(n_rows, n_bad, replace=False)
            corrupted[col] = corrupted[col].astype(object)
            for idx in bad_idx:
                corrupted.iloc[idx, corrupted.columns.get_loc(col)] = rng.choice(
                    ["N/A", "unknown", "#REF!", "TBD", "-"]
                )

    # 3. Inject typos in categorical columns
    for col in domain_config.categorical_columns:
        if col in corrupted.columns:
            n_typos = max(1, int(n_rows * rate * 0.2))
            typo_idx = rng.choice(n_rows, n_typos, replace=False)
            for idx in typo_idx:
                val = str(corrupted.iloc[idx, corrupted.columns.get_loc(col)])
                corrupted.iloc[idx, corrupted.columns.get_loc(col)] = _add_typo(val, rng)

    # 4. Inject duplicates
    n_dupes = max(1, int(n_rows * rate * 0.1))
    dupe_idx = rng.choice(n_rows, n_dupes, replace=False)
    dupes = corrupted.iloc[dupe_idx].copy()
    corrupted = pd.concat([corrupted, dupes], ignore_index=True)

    # 5. Inject whitespace issues
    for col in domain_config.categorical_columns[:2]:
        if col in corrupted.columns:
            n_ws = max(1, int(n_rows * rate * 0.2))
            ws_idx = rng.choice(len(corrupted), n_ws, replace=False)
            for idx in ws_idx:
                val = str(corrupted.iloc[idx, corrupted.columns.get_loc(col)])
                corrupted.iloc[idx, corrupted.columns.get_loc(col)] = f"  {val}  "

    return corrupted


def _add_typo(text: str, rng: np.random.Generator) -> str:
    """Add a realistic typo to a string."""
    if len(text) < 2:
        return text
    typo_type = rng.choice(["swap", "delete", "insert", "case"])
    idx = rng.integers(0, len(text))
    if typo_type == "swap" and idx < len(text) - 1:
        return text[:idx] + text[idx + 1] + text[idx] + text[idx + 2:]
    elif typo_type == "delete":
        return text[:idx] + text[idx + 1:]
    elif typo_type == "insert":
        char = rng.choice(list(string.ascii_lowercase))
        return text[:idx] + char + text[idx:]
    else:
        return text[:idx] + text[idx].swapcase() + text[idx + 1:]


def compute_dq_score(df: pd.DataFrame, domain_config: DomainConfig) -> dict:
    """Compute data quality metrics: completeness, consistency, uniqueness, overall."""
    available_cols = [c for c in domain_config.columns if c in df.columns]

    # Completeness: 1 - (null ratio)
    if available_cols:
        null_ratio = df[available_cols].isnull().sum().sum() / (len(df) * len(available_cols))
        completeness = 1.0 - null_ratio
    else:
        completeness = 1.0

    # Consistency: check type correctness for numeric columns
    consistency_scores = []
    for col in domain_config.numeric_columns:
        if col in df.columns:
            valid = df[col].apply(lambda x: _is_numeric(x)).mean()
            consistency_scores.append(valid)
    consistency = float(np.mean(consistency_scores)) if consistency_scores else 1.0

    # Uniqueness: 1 - (duplicate ratio)
    if len(df) > 0:
        n_dupes = df.duplicated(subset=available_cols[:5], keep='first').sum()
        uniqueness = 1.0 - (n_dupes / len(df))
    else:
        uniqueness = 1.0

    overall = 0.40 * completeness + 0.35 * consistency + 0.25 * uniqueness

    return {
        "completeness": round(completeness, 4),
        "consistency": round(consistency, 4),
        "uniqueness": round(uniqueness, 4),
        "overall": round(overall, 4),
    }


def _is_numeric(val) -> bool:
    """Check if a value is numeric (or null, which is valid)."""
    if pd.isna(val):
        return True
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def compute_dq_score_with_lfs(df: pd.DataFrame, domain: str,
                               lfs: list) -> float:
    """Compute DQ score using Snorkel-style labeling functions with majority vote."""
    if not lfs or len(df) == 0:
        config = DOMAINS.get(domain)
        if config:
            return compute_dq_score(df, config)["overall"]
        return 0.5

    ABSTAIN, BAD, GOOD = -1, 0, 1
    row_scores = []

    for _, row in df.iterrows():
        votes = []
        for lf in lfs:
            try:
                vote = lf(row)
                if vote != ABSTAIN:
                    votes.append(vote)
            except Exception:
                continue
        if votes:
            row_scores.append(sum(v == GOOD for v in votes) / len(votes))
        else:
            row_scores.append(0.5)

    return float(np.mean(row_scores))


def format_preview(df: pd.DataFrame, n: int = 5) -> str:
    """Format first n rows as a text table."""
    return df.head(n).to_string(index=False, max_colwidth=30)


def format_columns_info(df: pd.DataFrame, domain_config: DomainConfig) -> str:
    """Format column info: name, dtype, null count."""
    lines = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        dtype = str(df[col].dtype)
        expected = "expected" if col in domain_config.columns else "extra"
        lines.append(f"{col}: {dtype}, nulls={null_count} ({expected})")
    return "\n".join(lines)
