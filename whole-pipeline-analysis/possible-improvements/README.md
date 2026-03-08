# Possible Improvements for DataSage

Research-backed proposals for improving DataSage environments, training, and reward design. Organized by pipeline stage + cross-cutting concerns.

## Context

DataSage is a 3-stage enterprise data pipeline (Clean → Enrich → Answer) trained with GRPO on Qwen 2.5-3B. Mentor feedback highlighted two key areas:

1. **Focus on environments** — Make environments more challenging, novel, and harder to solve with simple heuristics
2. **Propagate business value to upstream layers** — The answering stage captures business value (persona-aware answers), but cleaning and enrichment optimize for proxies (DQ score, coverage count) disconnected from downstream utility

## The Gap

The original design doc specified `downstream_signal` in reward formulas:
- Cleaning: `0.70 * dq_score + 0.30 * downstream_signal`
- Enrichment: `0.50 * coverage + 0.50 * downstream_signal`

But the current implementation dropped it:
- Cleaning: `0.50 * dq_after + 0.50 * improvement_delta` (no downstream signal)
- Enrichment: `coverage` (no downstream signal at all)

This is the core issue: upstream stages don't know whether their actions actually help answer business questions.

## Folder Structure

| Folder | Focus | Key Improvement |
|--------|-------|-----------------|
| [`cleaning/`](cleaning/) | Cleaning environment | Dependent corruptions, step cost, schema drift |
| [`enrichment/`](enrichment/) | Enrichment environment | Budget constraints, conflicting sources, utility-aware selection |
| [`answering/`](answering/) | Answering environment | Multi-turn dialogue, adversarial questions, iterative refinement |
| [`pipeline/`](pipeline/) | Cross-stage reward propagation | Downstream signal, HER relabeling, LLM-as-judge |
| [`grpo-training/`](grpo-training/) | GRPO algorithm improvements | GDPO, λ-GRPO, curriculum learning, turn-level credit |

## Priority Matrix

| Improvement | Impact | Effort | Judge Appeal (40% env innovation) |
|-------------|--------|--------|-----------------------------------|
| **Downstream reward signal** (pipeline/) | Very High | Low | High — shows E2E thinking |
| **Step cost + invalid op penalty** (cleaning/) | High | Very Low | Medium — makes env harder |
| **Budget-constrained enrichment** (enrichment/) | High | Low | High — real decision-making |
| **Multi-turn answering** (answering/) | Very High | Medium | Very High — transforms env into true RL |
| **Schema drift** (cleaning/) | High | Medium | Very High — Patronus AI prize alignment |
| **GDPO multi-reward normalization** (grpo-training/) | High | Very Low | Medium — training improvement |
| **Curriculum learning** (grpo-training/) | Medium | Medium | Medium — training improvement |
| **Dependent corruptions** (cleaning/) | High | Medium | High — planning required |

## How to Read This

Each subfolder contains:
- `README.md` — Summary of proposed improvements with concrete code sketches
- `papers.md` — Relevant academic papers with arxiv links and applicability notes

All proposals are designed to be **consistent with OpenEnv 0.2.1** (same action/observation patterns, HF Space deployment, GRPO training via Colab notebooks).
