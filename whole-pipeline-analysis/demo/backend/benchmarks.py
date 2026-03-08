"""Benchmark runner for DataSage models across all tasks and domains."""

import json
import os
import time
from .config import DOMAINS, PERSONAS, MODEL_GROUPS, MODELS
from .agent import run_episode


def run_datasage_benchmarks(n_episodes: int = 3, tasks: list = None,
                             domains: list = None, progress_callback=None) -> dict:
    """Run full DataSage benchmark suite across models, tasks, domains."""
    tasks = tasks or ["cleaning", "enrichment", "answering"]
    domains = domains or DOMAINS
    all_results = {}
    total_runs = 0
    completed = 0

    # Count total runs
    for task in tasks:
        models = MODEL_GROUPS.get(task, [])
        for _ in models:
            for _ in domains:
                if task == "answering":
                    total_runs += len(PERSONAS) * n_episodes
                else:
                    total_runs += n_episodes

    for task in tasks:
        task_results = {}
        models = MODEL_GROUPS.get(task, [])

        for model_key in models:
            model_results = {}

            for domain in domains:
                if task == "answering":
                    domain_results = {}
                    for persona in PERSONAS:
                        persona_metrics = []
                        for ep in range(n_episodes):
                            result = run_episode(task, domain, model_key, persona, seed=42 + ep)
                            persona_metrics.append(result.get("metrics", {}))
                            completed += 1
                            if progress_callback:
                                progress_callback(completed / total_runs,
                                                  f"{task}/{domain}/{persona}/{model_key} ep{ep+1}")
                        domain_results[persona] = _aggregate_metrics(persona_metrics)
                    model_results[domain] = domain_results
                else:
                    episode_metrics = []
                    for ep in range(n_episodes):
                        result = run_episode(task, domain, model_key, seed=42 + ep)
                        episode_metrics.append(result.get("metrics", {}))
                        completed += 1
                        if progress_callback:
                            progress_callback(completed / total_runs,
                                              f"{task}/{domain}/{model_key} ep{ep+1}")
                    model_results[domain] = _aggregate_metrics(episode_metrics)

            task_results[model_key] = model_results
        all_results[task] = task_results

    # Compute E2E scores
    all_results["e2e_scores"] = _compute_e2e(all_results)
    return all_results


def _aggregate_metrics(metrics_list: list) -> dict:
    """Aggregate metrics across episodes."""
    if not metrics_list:
        return {}

    agg = {}
    for key in metrics_list[0]:
        values = [m.get(key, 0) for m in metrics_list if isinstance(m.get(key), (int, float))]
        if values:
            agg[key] = round(sum(values) / len(values), 4)
            agg[f"{key}_std"] = round(
                (sum((v - agg[key]) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5, 4
            )
    agg["n_episodes"] = len(metrics_list)
    return agg


def _compute_e2e(results: dict) -> dict:
    """Compute end-to-end scores per model."""
    e2e = {}
    # Get unique model base names
    all_models = set()
    for task_results in results.values():
        if isinstance(task_results, dict):
            all_models.update(task_results.keys())

    for model_key in all_models:
        if model_key.startswith("e2e"):
            continue

        # Cleaning score
        cleaning_data = results.get("cleaning", {}).get(model_key, {})
        cleaning_scores = []
        for domain_data in cleaning_data.values():
            if isinstance(domain_data, dict) and "final_dq_score" in domain_data:
                cleaning_scores.append(domain_data["final_dq_score"])
        cleaning_avg = sum(cleaning_scores) / max(len(cleaning_scores), 1) if cleaning_scores else 0

        # Enrichment score
        enrichment_data = results.get("enrichment", {}).get(model_key, {})
        enrichment_scores = []
        for domain_data in enrichment_data.values():
            if isinstance(domain_data, dict) and "final_coverage" in domain_data:
                enrichment_scores.append(domain_data["final_coverage"])
        enrichment_avg = sum(enrichment_scores) / max(len(enrichment_scores), 1) if enrichment_scores else 0

        # Answering score
        answering_data = results.get("answering", {}).get(model_key, {})
        answering_scores = []
        for domain_data in answering_data.values():
            if isinstance(domain_data, dict):
                for persona_data in domain_data.values():
                    if isinstance(persona_data, dict) and "combined_score" in persona_data:
                        answering_scores.append(persona_data["combined_score"])
        answering_avg = sum(answering_scores) / max(len(answering_scores), 1) if answering_scores else 0

        e2e[model_key] = {
            "cleaning_avg": round(cleaning_avg, 4),
            "enrichment_avg": round(enrichment_avg, 4),
            "answering_avg": round(answering_avg, 4),
            "e2e_score": round(0.30 * cleaning_avg + 0.30 * enrichment_avg + 0.40 * answering_avg, 4),
        }

    return e2e


def save_results(results: dict, path: str = None):
    """Save benchmark results to JSON."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmark_results.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return path


def load_results(path: str = None) -> dict:
    """Load benchmark results from JSON."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'benchmark_results.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}
