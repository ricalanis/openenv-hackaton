"""Simple scripted baseline for the FSDS cleaning environment.

This is not a full RL trainer. It provides a local baseline that judges can compare
against learned policies and also helps you smoke-test the environment.
"""

from fsds_cleaning_env import FSDSCleaningEnv

SCRIPTED_POLICY = {
    "ecommerce_mobile": [
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
}


def main() -> None:
    with FSDSCleaningEnv(base_url="http://localhost:8000").sync() as env:
        env.reset(task_id="ecommerce_mobile")
        print(env.call_tool("get_task_brief"))
        print(env.call_tool("profile_data"))
        for operation, column in SCRIPTED_POLICY["ecommerce_mobile"]:
            kwargs = {"operation": operation}
            if column is not None:
                kwargs["column"] = column
            print(env.call_tool("apply_cleaning_operation", **kwargs))
        print(env.call_tool("run_quality_gates"))
        print(env.call_tool("submit_solution"))


if __name__ == "__main__":
    main()
