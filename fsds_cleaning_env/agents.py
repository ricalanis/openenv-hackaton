"""Baseline agents for the FSDS Cleaning Environment.

Provides RandomAgent, HeuristicAgent, and extensible Agent interface for
benchmarking, evaluation, and RL training.
"""

from __future__ import annotations

import random
from typing import Any, Protocol, TypedDict

from fsds_cleaning_env.server.cleaning_environment import AVAILABLE_OPERATIONS


class ToolCall(TypedDict):
    """A tool invocation: tool_name and arguments for env.call_tool."""

    tool: str
    arguments: dict[str, Any]

# Scripted policies derived from required_ops. Format: list of (operation, column).
HEURISTIC_POLICIES: dict[str, list[tuple[str, str | None]]] = {
    "ecommerce_mobile": [
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
    ],
    "subscription_churn": [
        ("replace_invalid_with_null", "monthly_spend"),
        ("replace_invalid_with_null", "age"),
        ("replace_invalid_with_null", "tenure_months"),
        ("replace_invalid_with_null", "payment_method"),
        ("cast_numeric", "age"),
        ("cast_numeric", "monthly_spend"),
        ("cast_numeric", "tenure_months"),
        ("impute_numeric", "age"),
        ("impute_numeric", "monthly_spend"),
        ("impute_numeric", "tenure_months"),
        ("clip_outliers_iqr", "monthly_spend"),
        ("normalize_categories", "plan_type"),
        ("normalize_categories", "payment_method"),
        ("impute_categorical", "plan_type"),
        ("impute_categorical", "payment_method"),
        ("drop_duplicates", None),
    ],
    "delivery_eta": [
        ("replace_invalid_with_null", "driver_rating"),
        ("replace_invalid_with_null", "late_deliveries_last_30d"),
        ("replace_invalid_with_null", "city"),
        ("replace_invalid_with_null", "vehicle_type"),
        ("cast_numeric", "driver_rating"),
        ("cast_numeric", "delivery_distance_km"),
        ("cast_numeric", "late_deliveries_last_30d"),
        ("impute_numeric", "driver_rating"),
        ("impute_numeric", "late_deliveries_last_30d"),
        ("impute_numeric", "delivery_distance_km"),
        ("clip_outliers_iqr", "delivery_distance_km"),
        ("normalize_categories", "city"),
        ("normalize_categories", "vehicle_type"),
        ("impute_categorical", "city"),
        ("impute_categorical", "vehicle_type"),
        ("drop_duplicates", None),
    ],
}


def _extract_reward(result: dict[str, Any]) -> float:
    if "reward" in result:
        return float(result["reward"])
    if "final_reward" in result:
        return float(result["final_reward"])
    return 0.0


class Agent(Protocol):
    """Protocol for agents that run episodes in the FSDS Cleaning Environment."""

    def run_episode(
        self,
        env: Any,
        task_id: str,
        max_steps: int = 18,
        seed: int | None = None,
        **reset_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run one episode and return the trajectory (list of step dicts with reward, result, tool_name)."""
        ...


class AgentWithAct(Protocol):
    """Agent that supports per-step action selection for RL and step-by-step control.

    Use act(observation, history) to get the next tool call; run_episode uses it in a loop.
    """

    def act(self, observation: dict[str, Any], history: list[dict[str, Any]]) -> ToolCall | None:
        """Return the next tool call, or None to submit and end the episode."""
        ...

    def run_episode(
        self,
        env: Any,
        task_id: str,
        max_steps: int = 18,
        seed: int | None = None,
        **reset_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run one episode by repeatedly calling act() until done or max_steps."""
        ...


class RandomAgent:
    """Uniform random over valid tool calls. Serves as a lower bound for evaluation."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def run_episode(
        self,
        env: Any,
        task_id: str,
        max_steps: int = 18,
        seed: int | None = None,
        **reset_kwargs: Any,
    ) -> list[dict[str, Any]]:
        trajectory: list[dict[str, Any]] = []
        reset_kwargs["seed"] = seed
        env.reset(task_id=task_id, **reset_kwargs)
        profile = env.call_tool("profile_data")
        columns = list(profile.get("columns", []))
        columns_no_target = [c for c in columns if c not in ("converted", "churned", "delivery_time_minutes")]

        steps = 0
        submitted = False
        while steps < max_steps and not submitted:
            action = self._rng.choice(["inspect", "clean", "gates", "submit"])
            if action == "inspect":
                tool_name = self._rng.choice(["profile_data", "preview_data", "get_task_brief"])
                result = env.call_tool(tool_name)
            elif action == "clean":
                tool_name = "apply_cleaning_operation"
                op = self._rng.choice(AVAILABLE_OPERATIONS)
                if op == "drop_duplicates":
                    result = env.call_tool(tool_name, operation=op)
                elif columns_no_target:
                    col = self._rng.choice(columns_no_target)
                    result = env.call_tool(tool_name, operation=op, column=col)
                else:
                    result = env.call_tool(tool_name, operation="drop_duplicates")
            elif action == "gates":
                tool_name = "run_quality_gates"
                result = env.call_tool(tool_name)
            else:
                tool_name = "submit_solution"
                result = env.call_tool(tool_name)
                submitted = result.get("done", False)
            trajectory.append({
                "tool_name": tool_name,
                "reward": _extract_reward(result),
                "result": result,
            })
            steps += 1

        return trajectory


class HeuristicAgent:
    """Rule-based agent that follows the canonical cleaning policy for each task."""

    def run_episode(
        self,
        env: Any,
        task_id: str,
        max_steps: int = 18,
        seed: int | None = None,
        **reset_kwargs: Any,
    ) -> list[dict[str, Any]]:
        policy = HEURISTIC_POLICIES.get(task_id, HEURISTIC_POLICIES["ecommerce_mobile"])
        reset_kwargs["seed"] = seed
        env.reset(task_id=task_id, **reset_kwargs)
        trajectory: list[dict[str, Any]] = []

        for operation, column in policy:
            if len(trajectory) >= max_steps:
                break
            kwargs: dict[str, Any] = {"operation": operation}
            if column is not None:
                kwargs["column"] = column
            result = env.call_tool("apply_cleaning_operation", **kwargs)
            trajectory.append({
                "tool_name": "apply_cleaning_operation",
                "reward": _extract_reward(result),
                "result": result,
            })

        if len(trajectory) < max_steps:
            result = env.call_tool("run_quality_gates")
            trajectory.append({
                "tool_name": "run_quality_gates",
                "reward": _extract_reward(result),
                "result": result,
            })
        if len(trajectory) < max_steps:
            result = env.call_tool("submit_solution")
            trajectory.append({
                "tool_name": "submit_solution",
                "reward": _extract_reward(result),
                "result": result,
            })

        return trajectory


def _default_parse_llm_output(text: str) -> ToolCall:
    """Parse JSON tool call from LLM output. Fallback to profile_data."""
    import json
    import re
    match = re.search(r"\{[^{}]*\"tool\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            d = json.loads(match.group())
            tool = d.get("tool", "profile_data")
            args = d.get("arguments", {})
            return {"tool": str(tool), "arguments": dict(args)}
        except (json.JSONDecodeError, TypeError):
            pass
    return {"tool": "profile_data", "arguments": {}}


class LLMAgentAdapter:
    """Adapter for HF/LLM-based agents. Wraps a callable that produces tool calls from context.

    Usage:
        def my_model_fn(observation, history) -> str:
            # Build prompt, call model, return raw text
            return model.generate(...)
        agent = LLMAgentAdapter(generate_fn=my_model_fn, parse_fn=parse_json_tool_call)
    """

    def __init__(
        self,
        generate_fn: Any = None,
        parse_fn: Any = None,
    ) -> None:
        self._generate_fn = generate_fn or (lambda obs, hist: '{"tool": "profile_data", "arguments": {}}')
        self._parse_fn = parse_fn or _default_parse_llm_output

    def act(self, observation: dict[str, Any], history: list[dict[str, Any]]) -> ToolCall | None:
        text = self._generate_fn(observation, history)
        return self._parse_fn(text)

    def run_episode(
        self,
        env: Any,
        task_id: str,
        max_steps: int = 18,
        seed: int | None = None,
        **reset_kwargs: Any,
    ) -> list[dict[str, Any]]:
        reset_kwargs["seed"] = seed
        env.reset(task_id=task_id, **reset_kwargs)
        trajectory: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        observation: dict[str, Any] = {}

        for _ in range(max_steps):
            tool_call = self.act(observation, history)
            if tool_call is None:
                result = env.call_tool("submit_solution")
                trajectory.append({"tool_name": "submit_solution", "reward": result.get("final_reward", 0.0), "result": result})
                break
            tool_name = tool_call["tool"]
            args = tool_call.get("arguments", {})
            result = env.call_tool(tool_name, **args)
            trajectory.append({"tool_name": tool_name, "reward": _extract_reward(result), "result": result})
            history.append({"observation": observation, "tool_call": tool_call, "result": result})
            observation = result
            if result.get("done", False):
                break

        return trajectory


SYSTEM_PROMPT = """\
You are a Data Cleaning Agent working in a Medallion data pipeline (Bronze → Silver).

Your job: inspect a dirty dataset and clean it to Silver quality by choosing \
the right tools in the right order.

## Methodology (FSDS + VDS)
1. INSPECT first: profile_data, preview_data, get_task_brief
2. CLEAN systematically: fix dtypes, strip whitespace, handle missing values, \
remove duplicates, clip outliers
3. VALIDATE before submitting: run_quality_gates to check quality gate
4. SUBMIT: submit_solution when all tests pass

## Output Format
Each turn, output exactly one JSON action:
{"tool": "<tool_name>", "arguments": {"operation": "<op>", "column": "<col_or_omit>"}}

Top-level tools: profile_data, preview_data, get_task_brief, run_quality_gates, submit_solution
Cleaning tool: apply_cleaning_operation — requires an "operation" argument.

Available operations for apply_cleaning_operation:
  drop_duplicates
  replace_invalid_with_null  (requires "column")
  cast_numeric               (requires "column")
  cast_datetime              (requires "column")
  impute_numeric             (requires "column"; optional "strategy": "median"|"mean")
  impute_categorical         (requires "column")
  normalize_categories       (requires "column")
  clip_outliers_iqr          (requires "column")

Examples:
  {"tool": "profile_data", "arguments": {}}
  {"tool": "apply_cleaning_operation", "arguments": {"operation": "drop_duplicates"}}
  {"tool": "apply_cleaning_operation", "arguments": {"operation": "cast_numeric", "column": "amount"}}
  {"tool": "run_quality_gates", "arguments": {}}
  {"tool": "submit_solution", "arguments": {}}

Think step by step. Inspect before cleaning. Validate before submitting."""


class LLMAgent:
    """Agent powered by a fine-tuned LLM checkpoint (Unsloth/HuggingFace).

    Loads the model once on first use and generates one JSON action per step
    conditioned on the current observation and episode history.

    Args:
        model_path: Path to the saved model directory (e.g. ``./data-cleaning-grpo-final``).
        max_new_tokens: Maximum tokens to generate per step.
        temperature: Sampling temperature (0.0 = greedy).
    """

    def __init__(
        self,
        model_path: str = "./data-cleaning-grpo-final",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        import json as _json
        from unsloth import FastLanguageModel  # type: ignore[import]

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._model = model
        self._tokenizer = tokenizer
        self._json = _json

    def _build_user_message(
        self, observation: dict[str, Any], history: list[dict[str, Any]]
    ) -> str:
        import json as _json
        parts: list[str] = []
        if not history:
            parts.append("You just received a dirty Bronze-layer dataset. What is your first action?")
        else:
            last = history[-1]
            obs_summary = _json.dumps(last["result"], ensure_ascii=False)[:400]
            parts.append(f"Last action: {last['tool_call']['tool']}")
            parts.append(f"Result (truncated): {obs_summary}")
            parts.append("What is your next action?")
        return "\n".join(parts)

    def _generate(self, observation: dict[str, Any], history: list[dict[str, Any]]) -> str:
        if self._model is None:
            self._load()

        import torch  # type: ignore[import]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_message(observation, history)},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def run_episode(
        self,
        env: Any,
        task_id: str,
        max_steps: int = 18,
        seed: int | None = None,
        **reset_kwargs: Any,
    ) -> list[dict[str, Any]]:
        reset_kwargs["seed"] = seed
        env.reset(task_id=task_id, **reset_kwargs)
        trajectory: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        observation: dict[str, Any] = {}

        for _ in range(max_steps):
            raw = self._generate(observation, history)
            tool_call = _default_parse_llm_output(raw)
            tool_name = tool_call["tool"]
            args = tool_call.get("arguments", {})
            result = env.call_tool(tool_name, **args)
            trajectory.append({
                "tool_name": tool_name,
                "reward": _extract_reward(result),
                "result": result,
                "raw_output": raw,
            })
            history.append({"observation": observation, "tool_call": tool_call, "result": result})
            observation = result
            if result.get("done", False):
                break

        return trajectory


__all__ = [
    "Agent",
    "AgentWithAct",
    "ToolCall",
    "RandomAgent",
    "HeuristicAgent",
    "LLMAgent",
    "LLMAgentAdapter",
    "HEURISTIC_POLICIES",
    "SYSTEM_PROMPT",
]
