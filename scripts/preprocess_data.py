#!/usr/bin/env python3
"""Pre-process evaluation data into docs/js/data.js and docs/data/training_curves.json."""

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DATA = os.path.join(BASE, "demo", "data")
DOCS = os.path.join(BASE, "docs")


def load(name):
    with open(os.path.join(DEMO_DATA, name)) as f:
        return json.load(f)


def main():
    eval_data = load("evaluation_results.json")
    bench = load("benchmark_results.json")
    extended = load("extended_results.json")

    ds = eval_data["datasage_finetuned"]
    base = eval_data["base_model"]
    meta = eval_data["metadata"]
    wt = eval_data["wandb_training"]

    # ── 1. Training curves → docs/data/training_curves.json ──
    # W&B labels are SWAPPED:
    #   "cleaning" key → has answering reward functions
    #   "answering" key → has cleaning reward functions
    # We map by reward function names, not JSON keys.

    curves = {}
    label_map = {
        "cleaning": "answering",   # JSON key "cleaning" is actually answering task
        "enrichment": "enrichment",
        "answering": "cleaning",   # JSON key "answering" is actually cleaning task
    }

    for json_key, actual_task in label_map.items():
        task_metrics = wt[json_key]["metrics"]
        epochs_raw = task_metrics["train/epoch"]["values"]
        reward_raw = task_metrics["train/reward"]["values"]
        loss_raw = task_metrics["train/loss"]["values"]
        kl_raw = task_metrics["train/kl"]["values"]
        reward_std_raw = task_metrics["train/reward_std"]["values"]

        # Find the last epoch reset (data has two concatenated W&B runs).
        # Keep only the second (most recent) run.
        reset_idx = 0
        for i in range(1, len(epochs_raw)):
            if epochs_raw[i] < epochs_raw[i - 1] - 0.5:
                reset_idx = i

        # Use the shortest array length after the reset as reference
        n = min(
            len(epochs_raw) - reset_idx,
            len(reward_raw) - reset_idx,
            len(loss_raw) - reset_idx,
            len(kl_raw) - reset_idx,
            len(reward_std_raw) - reset_idx,
        )

        epochs = epochs_raw[reset_idx : reset_idx + n]
        # Round epochs for cleaner labels
        epochs = [round(e, 3) for e in epochs]

        curve = {
            "steps": list(range(n)),  # sequential index
            "epochs": epochs,
            "reward": reward_raw[reset_idx : reset_idx + n],
            "loss": loss_raw[reset_idx : reset_idx + n],
            "kl": kl_raw[reset_idx : reset_idx + n],
            "reward_std": reward_std_raw[reset_idx : reset_idx + n],
            "component_rewards": {},
        }

        # Extract individual reward functions (aligned to same range)
        for mk, mv in task_metrics.items():
            if mk.startswith("train/rewards/") and mk.endswith("/mean"):
                reward_name = mk.replace("train/rewards/", "").replace("/mean", "")
                vals = mv["values"]
                curve["component_rewards"][reward_name] = vals[reset_idx : reset_idx + min(n, len(vals) - reset_idx)]

        curves[actual_task] = curve
        print(f"  {actual_task}: {n} points (epoch {epochs[0]:.2f}→{epochs[-1]:.2f}), reset_idx={reset_idx}")

    curves_path = os.path.join(DOCS, "data", "training_curves.json")
    with open(curves_path, "w") as f:
        json.dump(curves, f)
    print(f"Wrote {curves_path} ({os.path.getsize(curves_path) / 1024:.1f} KB)")

    # Also write as JS for local file:// usage (no fetch needed)
    curves_js_path = os.path.join(DOCS, "js", "training_curves.js")
    with open(curves_js_path, "w") as f:
        f.write("// Auto-generated training curves — do not edit manually\n")
        f.write(f"const TRAINING_CURVES = {json.dumps(curves)};\n")
    print(f"Wrote {curves_js_path} ({os.path.getsize(curves_js_path) / 1024:.1f} KB)")

    # ── 2. Inline data → docs/js/data.js ──

    # DataSage vs Base comparison
    datasage_vs_base = {
        "datasage": {
            "cleaning": round(ds["cleaning"]["avg_reward_mean"], 4),
            "enrichment": round(ds["enrichment"]["avg_reward_mean"], 4),
            "answering": round(ds["answering"]["reward_mean"], 4),
        },
        "base_qwen": {
            "cleaning": round(base["cleaning"]["avg_reward_mean"], 4),
            "enrichment": round(base["enrichment"]["avg_reward_mean"], 4),
            "answering": round(base["answering"]["reward_mean"], 4),
        },
        "improvement": {
            "cleaning": round(
                (ds["cleaning"]["avg_reward_mean"] - base["cleaning"]["avg_reward_mean"])
                / base["cleaning"]["avg_reward_mean"]
                * 100,
                1,
            ),
            "enrichment": round(
                (ds["enrichment"]["avg_reward_mean"] - base["enrichment"]["avg_reward_mean"])
                / max(base["enrichment"]["avg_reward_mean"], 0.001)
                * 100,
                1,
            ),
            "answering": round(
                (ds["answering"]["reward_mean"] - base["answering"]["reward_mean"])
                / base["answering"]["reward_mean"]
                * 100,
                1,
            ),
        },
    }

    # GPT-4o-mini vs Qwen3-8B from real benchmarks
    real = bench["real_results"]
    benchmark_comparison = {
        "gpt4o_mini": {
            "cleaning": round(real["cleaning"]["gpt-4o-mini"]["avg_reward_mean"], 4),
            "enrichment": round(real["enrichment"]["gpt-4o-mini"]["avg_reward_mean"], 4),
            "answering": round(real["answering"]["gpt-4o-mini"]["reward_mean"], 4),
        },
        "qwen3_8b": {
            "cleaning": round(real["cleaning"]["qwen3-8b"]["avg_reward_mean"], 4),
            "enrichment": round(real["enrichment"]["qwen3-8b"]["avg_reward_mean"], 4),
            "answering": round(real["answering"]["qwen3-8b"]["reward_mean"], 4),
        },
    }

    # Per-domain answering breakdown (DataSage)
    per_domain_answering = {}
    for model_key, model_data, model_label in [
        ("datasage", ds, "DataSage LoRA"),
        ("base_qwen", base, "Base Qwen2.5-3B"),
    ]:
        per_domain_answering[model_key] = {}
        for domain, ddata in model_data["answering"]["per_domain"].items():
            per_domain_answering[model_key][domain] = round(ddata["reward_mean"], 4)

    # Per-persona answering (aggregate from episodes)
    def aggregate_by_persona(episodes):
        buckets = {}
        for ep in episodes:
            p = ep.get("persona", "unknown")
            buckets.setdefault(p, []).append(ep["reward"])
        return {p: round(sum(v) / len(v), 4) for p, v in buckets.items()}

    per_persona = {
        "datasage": aggregate_by_persona(ds["answering"]["per_episode"]),
        "base_qwen": aggregate_by_persona(base["answering"]["per_episode"]),
    }

    # Also aggregate benchmark models by persona
    for model_key, ext_key in [("gpt4o_mini", "gpt-4o-mini"), ("qwen3_8b", "qwen3-8b")]:
        per_persona[model_key] = aggregate_by_persona(
            real["answering"][ext_key]["per_episode"]
        )

    # Also aggregate benchmark models by domain
    for model_key, ext_key in [("gpt4o_mini", "gpt-4o-mini"), ("qwen3_8b", "qwen3-8b")]:
        dom_buckets = {}
        for ep in real["answering"][ext_key]["per_episode"]:
            d = ep["domain"]
            dom_buckets.setdefault(d, []).append(ep["reward"])
        per_domain_answering[model_key] = {
            d: round(sum(v) / len(v), 4) for d, v in dom_buckets.items()
        }

    # Heatmap data: DataSage per-domain per-task
    heatmap = {
        "cleaning": {},
        "enrichment": {},
        "answering": {},
    }
    for domain in ["hr", "sales", "pm", "it_ops"]:
        if domain in ds["cleaning"]["per_domain"]:
            heatmap["cleaning"][domain] = round(
                ds["cleaning"]["per_domain"][domain]["avg_reward_mean"], 4
            )
        if domain in ds["enrichment"]["per_domain"]:
            heatmap["enrichment"][domain] = round(
                ds["enrichment"]["per_domain"][domain]["avg_reward_mean"], 4
            )
        if domain in ds["answering"]["per_domain"]:
            heatmap["answering"][domain] = round(
                ds["answering"]["per_domain"][domain]["reward_mean"], 4
            )

    # Q&A showcase — pick best answering episodes from extended results
    qa_showcase = []
    for model_key, ext_key in [("qwen3-8b", "qwen3-8b")]:
        episodes = extended["answering_extended"][ext_key]["episodes"]
        sorted_eps = sorted(episodes, key=lambda x: x["reward"], reverse=True)
        for ep in sorted_eps[:2]:
            qa_showcase.append(
                {
                    "model": ext_key,
                    "domain": ep["domain"],
                    "persona": ep["persona"],
                    "question": ep["question"],
                    "answer": ep["answer"][:300] + "..." if len(ep["answer"]) > 300 else ep["answer"],
                    "cited_columns": ep["cited_columns"],
                    "reward": ep["reward"],
                }
            )

    # Training config summary
    training_config = {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "quantization": "4-bit (BnB)",
        "lora_r": 16,
        "lora_alpha": 16,
        "optimizer": "AdamW 8-bit",
        "epochs": 3,
        "beta": 0.001,
        "epsilon": 0.2,
        "loss_type": "BNPO",
        "max_steps_per_task": 192,
    }

    # HF Spaces use huggingface.co/spaces/ URLs (not .hf.space)
    hf_spaces = {
        "cleaning": "https://huggingface.co/spaces/ricalanis/datasage-cleaning",
        "enrichment": "https://huggingface.co/spaces/ricalanis/datasage-enrichment",
        "answering": "https://huggingface.co/spaces/ricalanis/datasage-answering",
    }

    # Environment details
    environments = {
        "cleaning": {
            "name": "Data Cleaning",
            "color": "#F59E0B",
            "description": "Detects and fixes data quality issues — missing values, duplicates, type errors",
            "reward_functions": ["cleaning_env_reward", "cleaning_json_format_reward"],
            "metric": "Data Quality Score",
            "hf_space": hf_spaces["cleaning"],
            "lora_repo": meta["lora_repos"]["cleaning"],
            "datasage_reward": round(ds["cleaning"]["avg_reward_mean"], 3),
            "base_reward": round(base["cleaning"]["avg_reward_mean"], 3),
        },
        "enrichment": {
            "name": "Data Enrichment",
            "color": "#EF4444",
            "description": "Adds computed columns and derived features from existing data",
            "reward_functions": [
                "enrichment_env_reward",
                "enrichment_json_format_reward",
                "source_relevance_reward",
            ],
            "metric": "Coverage (fields added)",
            "hf_space": hf_spaces["enrichment"],
            "lora_repo": meta["lora_repos"]["enrichment"],
            "datasage_reward": round(ds["enrichment"]["avg_reward_mean"], 3),
            "base_reward": round(base["enrichment"]["avg_reward_mean"], 3),
        },
        "answering": {
            "name": "Data Answering",
            "color": "#3B82F6",
            "description": "Answers natural language questions grounded in the dataset",
            "reward_functions": [
                "answering_env_reward",
                "answering_json_format_reward",
                "patronus_reward_fn",
                "persona_match_reward",
            ],
            "metric": "Composite Reward",
            "hf_space": hf_spaces["answering"],
            "lora_repo": meta["lora_repos"]["answering"],
            "datasage_reward": round(ds["answering"]["reward_mean"], 3),
            "base_reward": round(base["answering"]["reward_mean"], 3),
        },
    }

    # Links
    links = {
        "github": "https://github.com/ricalanis/openenv-hackaton",
        "lora_repos": meta["lora_repos"],
        "hf_spaces": hf_spaces,
    }

    # Radar chart data (all 4 models, 3 axes)
    radar = {
        "datasage": [
            round(ds["cleaning"]["avg_reward_mean"], 3),
            round(ds["enrichment"]["avg_reward_mean"], 3),
            round(ds["answering"]["reward_mean"], 3),
        ],
        "base_qwen": [
            round(base["cleaning"]["avg_reward_mean"], 3),
            round(base["enrichment"]["avg_reward_mean"], 3),
            round(base["answering"]["reward_mean"], 3),
        ],
        "gpt4o_mini": [
            round(real["cleaning"]["gpt-4o-mini"]["avg_reward_mean"], 3),
            round(real["enrichment"]["gpt-4o-mini"]["avg_reward_mean"], 3),
            round(real["answering"]["gpt-4o-mini"]["reward_mean"], 3),
        ],
        "qwen3_8b": [
            round(real["cleaning"]["qwen3-8b"]["avg_reward_mean"], 3),
            round(real["enrichment"]["qwen3-8b"]["avg_reward_mean"], 3),
            round(real["answering"]["qwen3-8b"]["reward_mean"], 3),
        ],
    }

    # Assemble data.js
    data_obj = {
        "datasageVsBase": datasage_vs_base,
        "benchmarkComparison": benchmark_comparison,
        "perDomainAnswering": per_domain_answering,
        "perPersona": per_persona,
        "heatmap": heatmap,
        "qaShowcase": qa_showcase,
        "trainingConfig": training_config,
        "environments": environments,
        "links": links,
        "radar": radar,
    }

    data_js = "// Auto-generated from demo/data/*.json — do not edit manually\n"
    data_js += f"const DATA = {json.dumps(data_obj, indent=2)};\n"

    data_js_path = os.path.join(DOCS, "js", "data.js")
    with open(data_js_path, "w") as f:
        f.write(data_js)
    print(f"Wrote {data_js_path} ({os.path.getsize(data_js_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
