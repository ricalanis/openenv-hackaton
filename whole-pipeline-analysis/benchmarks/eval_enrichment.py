"""Evaluate enrichment model on coverage and information gain per domain."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.shared.domains import DOMAINS
from environments.shared.enterprise_data import load_domain_data
from environments.shared.enrichment_sources import get_available_enrichments, lookup


def eval_enrichment(model_name: str = None, domain: str = "hr", n_episodes: int = 10) -> dict:
    """Run enrichment evaluation for a domain.

    Returns dict of metrics: coverage_ratio, information_gain, fields_added_avg.
    """
    config = DOMAINS[domain]
    available = get_available_enrichments(domain)

    coverage_ratios = []
    info_gains = []

    for ep in range(n_episodes):
        df = load_domain_data(domain, sample_size=50)
        initial_cols = len(df.columns)

        if model_name:
            # TODO: Load model and run inference
            fields_added = available  # Upper bound
        else:
            # Baseline: add first 2 enrichments only
            fields_added = available[:2]

        # Apply enrichments
        for source in fields_added:
            values = []
            for _, row in df.iterrows():
                val = lookup(domain, source, row.to_dict())
                values.append(val)
            df[source] = values

        final_cols = len(df.columns)
        coverage = len(fields_added) / max(len(available), 1)
        info_gain = (final_cols - initial_cols) / max(initial_cols, 1)

        coverage_ratios.append(coverage)
        info_gains.append(info_gain)

    return {
        "coverage_ratio": sum(coverage_ratios) / len(coverage_ratios),
        "information_gain": sum(info_gains) / len(info_gains),
        "fields_added_avg": sum(len(available) for _ in range(n_episodes)) / n_episodes if model_name else 2.0,
        "possible_enrichments": len(available),
        "episodes": n_episodes,
    }


if __name__ == "__main__":
    for domain in DOMAINS:
        m = eval_enrichment(domain=domain)
        print(f"{domain}: {m}")
