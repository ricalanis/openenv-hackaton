from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from fsds_cleaning_env import FSDSCleaningEnv  # noqa: E402


SCRIPTED_POLICY = [
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


def run_scripted_episode(base_url: str = "http://localhost:8000") -> dict:
    with FSDSCleaningEnv(base_url=base_url).sync() as env:
        env.reset(task_id="ecommerce_mobile")
        for operation, column in SCRIPTED_POLICY:
            kwargs = {"operation": operation}
            if column is not None:
                kwargs["column"] = column
            env.call_tool("apply_cleaning_operation", **kwargs)
        gates = env.call_tool("run_quality_gates")
        final = env.call_tool("submit_solution")
    return {"quality_gates": gates, "final": final}


def main() -> None:
    """Run two identical scripted episodes and compare outcomes.

    This is a lightweight determinism check for the environment when given
    the same sequence of tool calls.
    """

    first = run_scripted_episode()
    second = run_scripted_episode()

    print("=== FIRST EPISODE (summary) ===")
    pprint(first["final"])
    print("\n=== SECOND EPISODE (summary) ===")
    pprint(second["final"])

    same_final_reward = first["final"].get("final_reward") == second["final"].get("final_reward")
    same_passed = first["final"].get("passed") == second["final"].get("passed")
    same_required_cov = first["final"].get("required_operation_coverage") == second["final"].get(
        "required_operation_coverage"
    )

    print("\n=== DETERMINISM CHECK ===")
    print(f"Final reward equal: {same_final_reward}")
    print(f"Pass/fail equal: {same_passed}")
    print(f"Required-operation coverage equal: {same_required_cov}")

    if all([same_final_reward, same_passed, same_required_cov]):
        print("Result: outcomes are deterministic for this scripted policy.")
    else:
        print("Result: outcomes differ; investigate potential non-determinism.")


if __name__ == "__main__":
    main()

