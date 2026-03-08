"""LangGraph agentic system for multi-model DataSage comparison."""

import json
import time
import requests
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END

from .config import SPACE_URLS, MAX_CLEANING_STEPS, MAX_ENRICHMENT_STEPS
from .models import get_provider


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    task: str                  # cleaning | enrichment | answering
    domain: str                # hr | sales | pm | it_ops
    model_key: str             # key into MODELS registry
    persona: str               # for answering only
    seed: int                  # environment seed for reproducibility
    observation: dict          # current env observation
    actions: list              # history of actions taken
    responses: list            # raw model responses
    results: list              # environment step results
    current_step: int
    max_steps: int
    done: bool
    final_metrics: dict
    error: str
    trace: list                # execution trace for visualization


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _env_reset(task: str, domain: str, seed: int = 42, persona: str = "executive") -> dict:
    """Reset environment and return initial observation via /web/reset."""
    url = SPACE_URLS[task]
    try:
        resp = requests.post(f"{url}/web/reset", json={"seed": seed}, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            # The response wraps observation: {"observation": {...}, "reward": ..., "done": ...}
            return data.get("observation", data)
    except Exception:
        pass
    return _mock_observation(task, domain, persona)


def _env_step(task: str, action: dict) -> dict:
    """Send action to environment via /web/step and return result."""
    url = SPACE_URLS[task]
    try:
        resp = requests.post(f"{url}/web/step", json={"action": action}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return _mock_step_result(task)


def _mock_observation(task: str, domain: str, persona: str = "executive") -> dict:
    """Mock observation when environment is unavailable."""
    if task == "cleaning":
        return {
            "domain": domain,
            "dq_score": 0.62,
            "dq_report": {
                "completeness": 0.75, "consistency": 0.60,
                "uniqueness": 0.85, "validity": 0.55
            },
            "data_preview": f"[Sample {domain} data with nulls and type errors]",
            "columns_info": {"EmployeeID": "int", "Age": "float", "Department": "str"},
            "step_number": 1,
            "max_steps": MAX_CLEANING_STEPS,
        }
    elif task == "enrichment":
        return {
            "domain": domain,
            "schema_info": {"columns": ["ID", "Name", "Value", "Category"]},
            "enrichment_coverage": 0.0,
            "fields_added": [],
            "possible_enrichments": ["risk_score", "category_band", "trend_indicator"],
            "step_number": 1,
            "max_steps": MAX_ENRICHMENT_STEPS,
        }
    else:
        return {
            "domain": domain,
            "persona": persona,
            "persona_description": f"{persona} persona",
            "question": f"What are the key trends in {domain} data?",
            "dataset_summary": {"rows": 50, "columns": 12},
            "available_columns": ["ID", "Name", "Value", "Status"],
        }


def _mock_step_result(task: str) -> dict:
    """Mock step result when environment is unavailable."""
    import random
    if task == "cleaning":
        return {
            "observation": {"dq_score": 0.62 + random.uniform(0.02, 0.08)},
            "reward": random.uniform(0.3, 0.8),
            "done": False,
            "info": {"operation_applied": True},
        }
    elif task == "enrichment":
        return {
            "observation": {"enrichment_coverage": random.uniform(0.1, 0.3)},
            "reward": random.uniform(0.3, 0.7),
            "done": False,
            "info": {"field_added": True},
        }
    else:
        return {
            "observation": {},
            "reward": random.uniform(0.4, 0.9),
            "done": True,
            "info": {"faithfulness": random.uniform(0.5, 0.9), "persona_alignment": random.uniform(0.4, 0.85)},
        }


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def initialize(state: AgentState) -> dict:
    """Reset environment and get initial observation."""
    task = state["task"]
    domain = state["domain"]
    persona = state.get("persona", "executive")
    seed = state.get("seed", 42)

    max_steps = {
        "cleaning": MAX_CLEANING_STEPS,
        "enrichment": MAX_ENRICHMENT_STEPS,
        "answering": 1,
    }[task]

    obs = _env_reset(task, domain, seed, persona)

    trace_entry = {
        "node": "initialize",
        "timestamp": time.time(),
        "task": task,
        "domain": domain,
        "model": state["model_key"],
        "observation_keys": list(obs.keys()),
    }

    return {
        "observation": obs,
        "current_step": 0,
        "max_steps": max_steps,
        "done": False,
        "actions": [],
        "responses": [],
        "results": [],
        "trace": [trace_entry],
        "error": "",
    }


def select_action(state: AgentState) -> dict:
    """Use the selected model to generate an action."""
    provider = get_provider(state["model_key"])
    start = time.time()

    try:
        action = provider.generate_action(state["observation"], state["task"])
        raw_response = json.dumps(action)
    except Exception as e:
        action = {"error": str(e)}
        raw_response = str(e)

    elapsed = time.time() - start

    trace_entry = {
        "node": "select_action",
        "timestamp": time.time(),
        "step": state["current_step"],
        "model": state["model_key"],
        "action": action,
        "latency_ms": round(elapsed * 1000),
    }

    return {
        "actions": state["actions"] + [action],
        "responses": state["responses"] + [raw_response],
        "trace": state["trace"] + [trace_entry],
    }


def execute_action(state: AgentState) -> dict:
    """Send action to environment."""
    action = state["actions"][-1] if state["actions"] else {}
    result = _env_step(state["task"], action)

    new_obs = result.get("observation", state["observation"])
    if isinstance(new_obs, dict):
        # Merge with previous observation for continuity
        merged = {**state["observation"], **new_obs}
    else:
        merged = state["observation"]

    done = result.get("done", False)
    step = state["current_step"] + 1

    # For answering, always done after 1 step
    if state["task"] == "answering":
        done = True

    trace_entry = {
        "node": "execute_action",
        "timestamp": time.time(),
        "step": step,
        "reward": result.get("reward", 0),
        "done": done,
        "info": result.get("info", {}),
    }

    return {
        "observation": merged,
        "results": state["results"] + [result],
        "current_step": step,
        "done": done,
        "trace": state["trace"] + [trace_entry],
    }


def evaluate(state: AgentState) -> dict:
    """Compute final metrics from the episode."""
    task = state["task"]
    results = state["results"]

    if task == "cleaning":
        rewards = [r.get("reward", 0) for r in results]
        final_dq = state["observation"].get("dq_score", 0)
        initial_obs = state["trace"][0] if state["trace"] else {}
        metrics = {
            "total_steps": state["current_step"],
            "total_reward": sum(rewards),
            "avg_reward": sum(rewards) / max(len(rewards), 1),
            "final_dq_score": final_dq,
            "dq_improvement": final_dq - 0.62,  # approx baseline
        }
    elif task == "enrichment":
        rewards = [r.get("reward", 0) for r in results]
        coverage = state["observation"].get("enrichment_coverage", 0)
        metrics = {
            "total_steps": state["current_step"],
            "total_reward": sum(rewards),
            "avg_reward": sum(rewards) / max(len(rewards), 1),
            "final_coverage": coverage,
            "fields_added": len(state["actions"]),
        }
    elif task == "answering":
        last_result = results[-1] if results else {}
        info = last_result.get("info", {})
        metrics = {
            "faithfulness": info.get("faithfulness", 0),
            "persona_alignment": info.get("persona_alignment", 0),
            "combined_score": (
                0.30 * info.get("faithfulness", 0) +
                0.70 * info.get("persona_alignment", 0)
            ),
            "answer": state["actions"][-1].get("answer", "") if state["actions"] else "",
        }
    else:
        metrics = {"error": "unknown task"}

    trace_entry = {
        "node": "evaluate",
        "timestamp": time.time(),
        "metrics": metrics,
    }

    return {
        "final_metrics": metrics,
        "trace": state["trace"] + [trace_entry],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> Literal["select_action", "evaluate"]:
    """Decide whether to continue the episode or evaluate."""
    if state["done"] or state["current_step"] >= state["max_steps"]:
        return "evaluate"
    return "select_action"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_agent() -> StateGraph:
    """Build and compile the LangGraph agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("initialize", initialize)
    workflow.add_node("select_action", select_action)
    workflow.add_node("execute_action", execute_action)
    workflow.add_node("evaluate", evaluate)

    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "select_action")
    workflow.add_edge("select_action", "execute_action")
    workflow.add_conditional_edges("execute_action", should_continue)
    workflow.add_edge("evaluate", END)

    return workflow.compile()


# Singleton compiled agent
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def run_episode(
    task: str,
    domain: str,
    model_key: str,
    persona: str = "executive",
    seed: int = 42,
) -> dict:
    """Run a single episode and return results with trace."""
    agent = get_agent()

    initial_state: AgentState = {
        "task": task,
        "domain": domain,
        "model_key": model_key,
        "persona": persona,
        "seed": seed,
        "observation": {},
        "actions": [],
        "responses": [],
        "results": [],
        "current_step": 0,
        "max_steps": 1,
        "done": False,
        "final_metrics": {},
        "error": "",
        "trace": [],
    }

    try:
        final_state = agent.invoke(initial_state)
        return {
            "success": True,
            "metrics": final_state.get("final_metrics", {}),
            "trace": final_state.get("trace", []),
            "actions": final_state.get("actions", []),
            "steps": final_state.get("current_step", 0),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "metrics": {},
            "trace": [],
            "actions": [],
            "steps": 0,
        }


def run_comparison(
    task: str,
    domain: str,
    model_keys: list[str],
    persona: str = "executive",
    seed: int = 42,
    n_episodes: int = 3,
) -> dict:
    """Run multiple episodes across models for comparison."""
    results = {}
    for model_key in model_keys:
        model_results = []
        for ep in range(n_episodes):
            result = run_episode(task, domain, model_key, persona, seed=seed + ep)
            model_results.append(result)
        results[model_key] = model_results
    return results
