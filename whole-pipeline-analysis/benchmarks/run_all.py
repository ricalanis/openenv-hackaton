"""Run all DataSage benchmarks and log to W&B."""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmarks.eval_cleaning import eval_cleaning
from benchmarks.eval_enrichment import eval_enrichment
from benchmarks.eval_answering import eval_answering
from environments.shared.domains import DOMAINS


def compute_e2e_score(cleaning_metrics, enrichment_metrics, answering_metrics) -> float:
    """Compute end-to-end pipeline score."""
    cleaning_avg = sum(m["dq_after"] for m in cleaning_metrics.values()) / len(cleaning_metrics)
    enrichment_avg = sum(m["coverage_ratio"] for m in enrichment_metrics.values()) / len(enrichment_metrics)
    answering_avg = sum(m["combined_score"] for m in answering_metrics.values()) / len(answering_metrics)
    return 0.30 * cleaning_avg + 0.30 * enrichment_avg + 0.40 * answering_avg


def run_all(model_cleaning=None, model_enrichment=None, model_answering=None,
            use_wandb=True, n_episodes=10):
    """Run all benchmarks across domains and personas."""

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="datasage", name="benchmark-suite")
        except Exception as e:
            print(f"W&B init failed: {e}, continuing without logging")
            use_wandb = False

    cleaning_metrics = {}
    enrichment_metrics = {}
    answering_metrics = {}

    for domain in DOMAINS:
        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")

        # Cleaning evaluation
        c_metrics = eval_cleaning(model_name=model_cleaning, domain=domain, n_episodes=n_episodes)
        cleaning_metrics[domain] = c_metrics
        print(f"  Cleaning: DQ {c_metrics['dq_before']:.4f} -> {c_metrics['dq_after']:.4f} (+{c_metrics['dq_improvement']:.4f})")

        if use_wandb:
            import wandb
            wandb.log({f"cleaning/{domain}/{k}": v for k, v in c_metrics.items()})

        # Enrichment evaluation
        e_metrics = eval_enrichment(model_name=model_enrichment, domain=domain, n_episodes=n_episodes)
        enrichment_metrics[domain] = e_metrics
        print(f"  Enrichment: coverage={e_metrics['coverage_ratio']:.4f}, info_gain={e_metrics['information_gain']:.4f}")

        if use_wandb:
            wandb.log({f"enrichment/{domain}/{k}": v for k, v in e_metrics.items()})

        # Answering evaluation (per persona)
        for persona in ["executive", "manager", "ic"]:
            a_metrics = eval_answering(model_name=model_answering, domain=domain,
                                       persona=persona, n_episodes=max(n_episodes // 2, 3))
            key = f"{domain}/{persona}"
            answering_metrics[key] = a_metrics
            print(f"  Answering ({persona}): faith={a_metrics['faithfulness']:.4f}, persona={a_metrics['persona_alignment']:.4f}, combined={a_metrics['combined_score']:.4f}")

            if use_wandb:
                wandb.log({f"answering/{domain}/{persona}/{k}": v
                          for k, v in a_metrics.items() if v is not None})

    # End-to-end score
    e2e = compute_e2e_score(cleaning_metrics, enrichment_metrics, answering_metrics)
    print(f"\n{'='*60}")
    print(f"End-to-End Score: {e2e:.4f}")
    print(f"{'='*60}")

    if use_wandb:
        wandb.log({"pipeline/end_to_end_score": e2e})
        wandb.finish()

    return {
        "cleaning": cleaning_metrics,
        "enrichment": enrichment_metrics,
        "answering": answering_metrics,
        "e2e_score": e2e,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DataSage benchmarks")
    parser.add_argument("--model-cleaning", default=None)
    parser.add_argument("--model-enrichment", default=None)
    parser.add_argument("--model-answering", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    run_all(
        model_cleaning=args.model_cleaning,
        model_enrichment=args.model_enrichment,
        model_answering=args.model_answering,
        use_wandb=not args.no_wandb,
        n_episodes=args.episodes,
    )
