# Whole Pipeline Analysis

End-to-end data science pipeline covering the full lifecycle: data preparation, environment design, model training, evaluation, inference, and deployment.

## Pipeline Stages

| Stage | Directory | Description |
|-------|-----------|-------------|
| **Data** | `data/` | Dataset fetching and labeling functions |
| **Environments** | `environments/` | RL environments for three core tasks: cleaning, enrichment, and answering |
| **Training** | `training/` | GRPO training scripts and Colab notebooks for each task stage |
| **Benchmarks** | `benchmarks/` | Evaluation suites (per-task + persona alignment) |
| **Inference** | `demo/inference/` | Step-by-step notebooks: W&B export, per-task inference, result aggregation |
| **Demo** | `demo/` | Gradio app with agent backend for interactive demonstration |
| **Deployment** | `deployment/` | Northflank configs and multi-stage training shell scripts |
| **Tests** | `tests/` | Unit and integration tests for environments, rewards, and notebooks |

## Additional Resources

- `tutorial/` — Setup guide and starter code for building custom environments
- `possible-improvements/` — Research notes and paper references per task area
- `docs/` — Design plans, environment analysis, and known issues
