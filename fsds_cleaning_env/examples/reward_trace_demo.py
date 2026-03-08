from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from fsds_cleaning_env import FSDSCleaningEnv  # noqa: E402


GOOD_POLICY = [
    ("replace_invalid_with_null", "country"),
    ("replace_invalid_with_null", "items_in_cart"),
    ("drop_duplicates", None),
    ("cast_numeric", "items_in_cart"),
    ("impute_numeric", "items_in_cart"),
    ("clip_outliers_iqr", "items_in_cart"),
    ("clip_outliers_iqr", "order_value"),
    ("normalize_categories", "device_os"),
    ("normalize_categories", "country"),
    ("cast_datetime", "event_date"),
]


BAD_POLICY = [
    # Intentionally minimal / misordered cleaning.
    ("drop_duplicates", None),
]


def run_policy(policy, base_url: str = "http://localhost:8000"):
    trace = []
    with FSDSCleaningEnv(base_url=base_url).sync() as env:
        env.reset(task_id="ecommerce_mobile")
        for op, col in policy:
            kwargs = {"operation": op}
            if col is not None:
                kwargs["column"] = col
            result = env.call_tool("apply_cleaning_operation", **kwargs)
            trace.append(
                {
                    "tool": "apply_cleaning_operation",
                    "operation": op,
                    "column": col,
                    "reward": result.get("reward", 0.0),
                    "quality_score": result.get("quality_score"),
                }
            )
        gates = env.call_tool("run_quality_gates")
        trace.append(
            {
                "tool": "run_quality_gates",
                "reward": gates.get("reward", 0.0),
                "passed": gates.get("passed"),
                "retention_ratio": gates.get("retention_ratio"),
            }
        )
        final = env.call_tool("submit_solution")
        trace.append(
            {
                "tool": "submit_solution",
                "reward": final.get("final_reward", 0.0),
                "passed": final.get("passed"),
                "required_operation_coverage": final.get("required_operation_coverage"),
                "cumulative_reward": final.get("cumulative_reward"),
            }
        )
    return trace


def main() -> None:
    good_trace = run_policy(GOOD_POLICY)
    bad_trace = run_policy(BAD_POLICY)

    print("=== GOOD POLICY REWARD TRACE ===")
    print(json.dumps(good_trace, indent=2))

    print("\n=== BAD POLICY REWARD TRACE ===")
    print(json.dumps(bad_trace, indent=2))


if __name__ == "__main__":
    main()

