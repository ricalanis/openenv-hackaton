"""Evaluate cleaning model on held-out data batches per domain."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.shared.domains import DOMAINS
from environments.shared.enterprise_data import load_domain_data, inject_corruption, compute_dq_score


def eval_cleaning(model_name: str = None, domain: str = "hr", n_episodes: int = 10) -> dict:
    """Run cleaning evaluation for a domain.

    If model_name is None, runs baseline (no model, random operations).
    Returns dict of metrics: dq_before, dq_after, dq_improvement, completeness, consistency, uniqueness.
    """
    config = DOMAINS[domain]
    metrics = {
        "dq_before": 0.0,
        "dq_after": 0.0,
        "dq_improvement": 0.0,
        "completeness": 0.0,
        "consistency": 0.0,
        "uniqueness": 0.0,
        "episodes": n_episodes,
    }

    dq_befores = []
    dq_afters = []
    completeness_scores = []
    consistency_scores = []
    uniqueness_scores = []

    for ep in range(n_episodes):
        # Load and corrupt data
        df = load_domain_data(domain, sample_size=50)
        corrupted = inject_corruption(df, config, rate=0.15)
        dq_before = compute_dq_score(corrupted, config)
        dq_befores.append(dq_before["overall"])

        if model_name:
            # TODO: Load model and run inference
            # For now, use gold standard as upper bound
            dq_after = compute_dq_score(df, config)
        else:
            # Baseline: no cleaning
            dq_after = dq_before

        dq_afters.append(dq_after["overall"])
        completeness_scores.append(dq_after["completeness"])
        consistency_scores.append(dq_after["consistency"])
        uniqueness_scores.append(dq_after["uniqueness"])

    metrics["dq_before"] = sum(dq_befores) / len(dq_befores)
    metrics["dq_after"] = sum(dq_afters) / len(dq_afters)
    metrics["dq_improvement"] = metrics["dq_after"] - metrics["dq_before"]
    metrics["completeness"] = sum(completeness_scores) / len(completeness_scores)
    metrics["consistency"] = sum(consistency_scores) / len(consistency_scores)
    metrics["uniqueness"] = sum(uniqueness_scores) / len(uniqueness_scores)

    return metrics


if __name__ == "__main__":
    for domain in DOMAINS:
        m = eval_cleaning(domain=domain)
        print(f"{domain}: {m}")
