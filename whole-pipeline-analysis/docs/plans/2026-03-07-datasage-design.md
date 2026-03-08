# DataSage Design

**Date:** 2026-03-07
**Status:** Approved
**Author:** ricalanis

## One-Liner

A 3-stage sequential RL environment where a Qwen model learns to clean, enrich, and answer HR data questions, with persona-aware rewards that weight each step's quality against downstream business validity.

## Target Prizes

| Target | Partner | Fit |
|---|---|---|
| Fleet AI | Scalable Oversight | 3 agents each overseen by metric agents |
| Scale AI | Long-horizon HR & IT workflows | Multi-step enterprise HR workflow |
| Main track | Statement 3.1 World Modeling | Real data pipeline as partially observable world |

## Architecture: Sequential Pipeline with Artifact Handoff

```
[HF Space 1: Cleaning Env] → [HF Dataset: cleaned] → Northflank Job 1 (GRPO)
[HF Space 2: Enrichment Env] → [HF Dataset: enriched] → Northflank Job 2 (GRPO)
[HF Space 3: Answering Env] → [HF Dataset: answers] → Northflank Job 3 (GRPO)
                                                      → Northflank Job 4 (Benchmarks)
All traced on W&B project "datasage"
```

## Project Structure

```
openenv-datasage/
├── environments/
│   ├── cleaning/          # HF Space: ricalanis/datasage-cleaning
│   ├── enrichment/        # HF Space: ricalanis/datasage-enrichment
│   └── answering/         # HF Space: ricalanis/datasage-answering
├── training/
│   ├── train_cleaning.py
│   ├── train_enrichment.py
│   ├── train_answering.py
│   └── shared/
├── benchmarks/
│   ├── eval_cleaning.py
│   ├── eval_enrichment.py
│   ├── eval_answering.py
│   └── run_all.py
├── orchestrator/
│   ├── pipeline.py
│   └── config.py
├── data/
│   └── fetch_datasets.py
├── deployment/
│   ├── northflank/
│   │   ├── job_stage1.yaml
│   │   ├── job_stage2.yaml
│   │   ├── job_stage3.yaml
│   │   ├── job_benchmark.yaml
│   │   └── pipeline.yaml
│   └── push_spaces.sh
└── README.md
```

## HuggingFace Artifacts

**Datasets:**
- `ricalanis/datasage-hr-raw` — IBM HR Attrition
- `ricalanis/datasage-hr-cleaned` — Stage 1 output
- `ricalanis/datasage-hr-enriched` — Stage 2 output
- `ricalanis/datasage-hr-answers` — Stage 3 output

**Models:**
- `ricalanis/datasage-qwen-cleaning`
- `ricalanis/datasage-qwen-enrichment`
- `ricalanis/datasage-qwen-answering`

**Spaces:**
- `ricalanis/datasage-cleaning`
- `ricalanis/datasage-enrichment`
- `ricalanis/datasage-answering`

## Environment Design

### Stage 1 — Cleaning

- **Observation:** Raw HR row(s) with nulls, typos, type mismatches + DQ report
- **Action:** JSON `{operation, column, value}` — fill_null, fix_type, remove_duplicate, standardize
- **Reward:** `0.70 * dq_score + 0.30 * downstream_signal`
- **Episode:** 10-row batch, done at DQ > 0.95 or max_steps=15

### Stage 2 — Enrichment

- **Observation:** Cleaned row(s) + schema + available sources
- **Action:** JSON `{operation, source, field_name, logic}` — add_field, lookup, compute_derived
- **Reward:** `0.50 * enrichment_coverage + 0.50 * downstream_signal`
- **Episode:** Batch, done at coverage > 0.80 or max_steps=12

### Stage 3 — Answering

- **Observation:** Enriched dataset summary + persona + question
- **Action:** JSON `{answer, cited_columns, reasoning}`
- **Reward:** `0.30 * faithfulness + 0.70 * persona_relevance`
- **Episode:** Single question, done after answer
- **Personas:** HRManager (operational), CFO (financial), Employee (plain language)

## Training Pipeline (Northflank)

Sequential Northflank Jobs on H100:
1. `train-cleaning` → push model → trigger Job 2
2. `train-enrichment` → push model → trigger Job 3
3. `train-answering` → push model → trigger Job 4
4. `run-benchmarks` → push results to W&B + HF

All use GRPO via TRL `rollout_func` pattern with `generate_rollout_completions`.

## Benchmark Tracing (W&B)

W&B project `datasage` with runs per stage:
- **cleaning:** dq/completeness, dq/consistency, dq/uniqueness, downstream_signal
- **enrichment:** coverage, info_gain, downstream_signal
- **answering:** ragas/faithfulness, ragas/relevance, ragas/recall, persona/align_score
- **benchmark-suite:** all metrics on held-out test, pipeline/end_to_end_score

## Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model | Qwen2.5-3B-Instruct | Trainable on H100 in hackathon time |
| Training | GRPO via TRL rollout_func | OpenEnv official pattern |
| vLLM | colocate mode | Single GPU, simpler |
| Downstream signal | Cached historical | No full pipeline during training |
| Enrichment data | Static CSVs (BLS, taxonomy) | Deterministic, fast |
| Personas | 3 archetypes | Minimum viable for demo |
| Benchmarks | Great Expectations + RAGAS + PersonaAlign | Industry-standard |
