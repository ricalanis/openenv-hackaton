from __future__ import annotations

import argparse
from pathlib import Path
import sys
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from fsds_cleaning_env.client import FSDSCleaningEnv
from openenv.core.env_server.mcp_types import CallToolAction


def print_step_result(label, result):
    print(f"=== {label} ===")
    obs = result.observation
    if hasattr(obs, "result") and obs.result is not None:
        print("Result:")
        pprint(obs.result)
    if hasattr(obs, "error") and obs.error is not None:
        print("Error:", obs.error)
    if hasattr(obs, "metadata") and obs.metadata:
        print("Metadata:", obs.metadata)
    print("Reward:", result.reward)
    print("Done:", result.done)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for the FSDS cleaning environment")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the running environment")
    args = parser.parse_args()

    env = FSDSCleaningEnv(base_url=args.base_url)

    with env.sync() as client:
        reset_result = client.reset(task_id="ecommerce_mobile")
        print("=== RESET ===")
        obs = reset_result.observation
        if hasattr(obs, "metadata") and obs.metadata:
            pprint(obs.metadata)
        print("Done:", reset_result.done)
        print()

        step_result = client.step(CallToolAction(tool_name="profile_data", arguments={}))
        print_step_result("PROFILE", step_result)
        if step_result.done:
            return

        for operation, column in [
            ("replace_invalid_with_null", "country"),
            ("replace_invalid_with_null", "items_in_cart"),
            ("replace_invalid_with_null", "device_os"),
            ("cast_numeric", "items_in_cart"),
            ("cast_numeric", "order_value"),
            ("impute_numeric", "items_in_cart"),
            ("impute_numeric", "order_value"),
            ("clip_outliers_iqr", "items_in_cart"),
            ("clip_outliers_iqr", "order_value"),
            ("normalize_categories", "device_os"),
            ("normalize_categories", "country"),
            ("impute_categorical", "device_os"),
            ("impute_categorical", "country"),
            ("cast_datetime", "event_date"),
            ("drop_duplicates", None),
        ]:
            args: dict = {"operation": operation}
            if column is not None:
                args["column"] = column
            step_result = client.step(
                CallToolAction(
                    tool_name="apply_cleaning_operation",
                    arguments=args,
                )
            )
            label = f"APPLY {operation}" + (f" ({column})" if column else "")
            print_step_result(label, step_result)
            if step_result.done:
                return

        step_result = client.step(CallToolAction(tool_name="run_quality_gates", arguments={}))
        print_step_result("QUALITY GATES", step_result)
        if step_result.done:
            return

        step_result = client.step(CallToolAction(tool_name="submit_solution", arguments={}))
        print_step_result("SUBMIT", step_result)


if __name__ == "__main__":
    main()
