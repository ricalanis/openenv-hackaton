---
title: DataSage — Whole Pipeline Analysis
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - rl
  - data-science
  - grpo
  - enterprise
  - end-to-end
---

# DataSage — Whole Pipeline Analysis

An end-to-end data science pipeline where LLM agents are trained via **GRPO** (Group Relative Policy Optimization) to perform enterprise data work across three sequential stages: **cleaning**, **enrichment**, and **persona-aligned question answering**.

> **New here?** Start with the [Tutorial](tutorial/SETUP_GUIDE.md) for environment setup, or jump to [Quick start](#quick-start) to run the full pipeline locally.

## What this pipeline covers

The agent learns to take raw, messy enterprise data and turn it into trustworthy, enriched datasets that power persona-aware analytical answers — all through reinforcement learning against live OpenEnv environments.

```
Raw Data (Kaggle/UCI/HF Hub)
     │
     ▼
  data/fetch_datasets.py  →  Parquet files + gold standards + DQ labels
     │
     ▼
  ┌─────────────────────────────────────────────────────┐
  │  Stage 1: Cleaning       (15 steps, DQ > 0.95)     │
  │  Stage 2: Enrichment     (12 steps, coverage > 0.80)│
  │  Stage 3: Answering      (1 step, persona-aligned)  │
  └─────────────────────────────────────────────────────┘
     │
     ▼
  Training (Unsloth + TRL GRPO on Qwen2.5-3B)
     │
     ▼
  Benchmarks → W&B logging → Demo (Gradio + LangGraph)
```

The pipeline is inspired by:
- **FSDS**: Bronze → Silver → Gold layers with DS Agent plus QA gates.
- **VDSAgents**: Explore-Agent + PCS-style unit tests and perturbation-minded validation.
- **Data Interpreter**: progressive, tool-driven, multi-step execution instead of one-shot code.

## Domains included

| Domain | Source | Rows | Key columns |
|--------|--------|------|-------------|
| **HR** | IBM HR Analytics (ODbL) | 1,470 | Age, Dept, JobRole, MonthlyIncome, Attrition |
| **Sales** | CRM Opportunities (Apache 2.0) | 8,800 | AccountName, Stage, Amount, Probability |
| **IT Ops** | UCI Incident Mgmt (CC BY 4.0) | 5,000 | Category, Priority, SLATarget, EscalationLevel |
| **PM** | Construction PM (CC0) | 1,000+ | Status, EstimatedHours, ActualHours, CompletionPct |

## Quick start

```bash
# 1. Install dependencies
pip install -e .

# 2. Fetch real-world datasets
python data/fetch_datasets.py

# 3. Start a local environment (e.g. cleaning)
uvicorn environments.cleaning.server.app:app --port 8000

# 4. Run the full benchmark suite
python benchmarks/run_all.py

# 5. Launch the demo app
python demo/app.py
```

## Pipeline stages

### Stage 1 — Cleaning

The agent turns a messy table into a trustworthy Silver table by profiling, deduplicating, fixing types, and passing quality gates.

| Tool | Description |
|------|-------------|
| `fill_null` | Fill nulls with median/mode/specific value |
| `fix_type` | Cast column to numeric |
| `remove_duplicate` | Drop duplicate rows |
| `standardize` | Normalize case + strip whitespace |
| `trim` | Strip leading/trailing whitespace |
| `correct_typo` | Replace wrong → correct value |

**Reward**: `0.50 × DQ_after + 0.50 × min((DQ_after − DQ_before) × 5, 1.0)` — rewards both absolute quality and improvement delta.

**DQ score**: `0.40 × completeness + 0.35 × consistency + 0.25 × uniqueness`, augmented with Snorkel-style labeling functions per domain.

### Stage 2 — Enrichment

The agent adds derived fields from a registry of 5 enrichment sources per domain (20 total) to maximize field coverage.

| Tool | Description |
|------|-------------|
| `add_field` | Add enrichment from a known source |
| `lookup` | External reference lookup |
| `compute_derived` | Compute metric from existing columns |
| `add_category` | Add categorical classification |

**Reward**: direct coverage ratio (fields added / total possible).

Examples: `salary_band` from MonthlyIncome, `flight_risk_score` from tenure + satisfaction + overtime, `velocity_score` from DaysInStage.

### Stage 3 — Answering

Single-step episode: the agent receives a domain, persona, question, and enriched dataset, then submits a persona-aligned analytical answer with citations.

**Personas**:
- **Executive** — strategic-financial language (revenue, ROI, trend, margin)
- **Manager** — operational-actionable language (team, performance, bottleneck, SLA)
- **IC** — plain-personal language (my, next step, deadline, assigned)

**Reward**: `0.30 × faithfulness + 0.70 × persona_alignment` (or `0.40 × patronus_score + 0.60 × persona_alignment` when Patronus Lynx is available).

## Training approach

**Base model**: `unsloth/Qwen2.5-3B-Instruct` (4-bit quantized)
**Method**: GRPO via TRL + Unsloth, LoRA adapters (r=16, alpha=16)
**Hardware**: H100 GPU, BF16, vLLM colocate mode

Each stage trains a separate LoRA adapter with stage-specific reward functions:

| Stage | Reward functions | Key technique |
|-------|-----------------|---------------|
| Cleaning | env_reward + json_format + reasoning quality | Direct prompt → completion |
| Enrichment | env_reward + source_relevance + json_format | `rollout_func` with live env observation injection |
| Answering | env_reward + patronus/faithfulness + json_format + persona_match | `rollout_func` with seeded env resets |

Stages 2 and 3 use `rollout_func` to inject real environment observations into the GRPO loop, eliminating context mismatch between training prompts and inference.

**Trained models** (HuggingFace Hub):
- `ricalanis/datasage-qwen-cleaning`
- `ricalanis/datasage-qwen-enrichment`
- `ricalanis/datasage-qwen-answering`

## Evaluation

```bash
# Run the full benchmark suite with W&B logging
python benchmarks/run_all.py

# Or individual evaluations
python benchmarks/eval_cleaning.py
python benchmarks/eval_enrichment.py
python benchmarks/eval_answering.py
python benchmarks/persona_align.py
```

**End-to-end score**: `0.30 × cleaning + 0.30 × enrichment + 0.40 × answering` — answering weighted highest as the most semantically demanding stage.

All metrics log to the `datasage` W&B project with per-domain and per-persona breakdowns.

## Inference notebooks

Colab-ready notebooks in `demo/inference/`, split to avoid OOM (one LoRA adapter per notebook):

| Notebook | Purpose |
|----------|---------|
| `00_wandb_export` | Export training curves from W&B |
| `01_cleaning_inference` | Run trained cleaning LoRA against live env |
| `02_enrichment_inference` | Run trained enrichment LoRA against live env |
| `03_answering_inference` | Run trained answering LoRA against live env |
| `04_aggregate_results` | Combine all results into final summary |

## Demo

The Gradio app (`demo/app.py`) provides five tabs:

| Tab | Content |
|-----|---------|
| Real Results | Live GPT-4o-mini and Qwen3-8B scores per episode/domain/persona |
| Full Comparison | Real + projected DataSage results with bar + radar charts |
| Standard Benchmarks | MMLU, HumanEval, GSM8K, ARC, HellaSwag, TruthfulQA scores |
| Live Agent Demo | Interactive: select task/model/episodes, run via LangGraph agent |
| Architecture | Diagrams of agent flow, multi-model switching, training pipeline |

The agent backend uses a LangGraph `StateGraph` (initialize → select_action → execute_action → evaluate) with pluggable model providers (GPT-4o-mini, Qwen3-8B via Fireworks).

## Deployment

### HF Spaces (environments)

```bash
# Deploy each environment as a Docker Space
openenv push --repo-id ricalanis/datasage-cleaning
openenv push --repo-id ricalanis/datasage-enrichment
openenv push --repo-id ricalanis/datasage-answering
```

Each space exposes `/web` (UI), `/docs` (OpenAPI), `/health`, and `/ws` (WebSocket for concurrent sessions).

### Northflank (training)

Three GPU jobs on `nf-gpu-hack-16-64` (16 vCPU, 64GB RAM, H100):

```bash
bash deployment/train_stage1.sh   # Cleaning GRPO
bash deployment/train_stage2.sh   # Enrichment GRPO
bash deployment/train_stage3.sh   # Answering GRPO
```

Each job: 2-hour active deadline, Docker image `nvcr.io/nvidia/pytorch:24.01-py3`, secrets injected for `HF_TOKEN` and `WANDB_API_KEY`.

## Use from a client

```python
from environments.cleaning import CleaningEnv
from environments.cleaning.models import CleaningAction

with CleaningEnv(base_url="https://ricalanis-datasage-cleaning.hf.space") as env:
    obs = env.reset()
    obs = env.step(CleaningAction(operation="fill_null", column="Age", value="median"))
    obs = env.step(CleaningAction(operation="remove_duplicate"))
    obs = env.step(CleaningAction(operation="fix_type", column="MonthlyIncome"))
```

## Project structure

```
whole-pipeline-analysis/
├── data/                    # Dataset fetching + Snorkel labeling functions
├── environments/
│   ├── shared/              # Domains, personas, enrichment sources, reward utils
│   ├── cleaning/            # Stage 1 environment (server + client)
│   ├── enrichment/          # Stage 2 environment (server + client)
│   └── answering/           # Stage 3 environment (server + client)
├── training/
│   ├── shared/              # Config + parsers shared across stages
│   ├── train_cleaning.py    # GRPO training for stage 1
│   ├── train_enrichment.py  # GRPO training for stage 2
│   ├── train_answering.py   # GRPO training for stage 3
│   └── ran/                 # Executed notebook outputs
├── benchmarks/              # Eval suites + persona alignment scorer
├── demo/
│   ├── app.py               # Gradio demo with 5 tabs
│   ├── backend/             # LangGraph agent + model providers
│   └── inference/           # 5 Colab inference notebooks
├── deployment/
│   ├── northflank/          # GPU job configs (JSON/YAML)
│   └── train_stage*.sh      # Training shell scripts
├── tests/                   # Unit + integration tests
├── tutorial/                # Setup guide + starter scaffolding
├── my_env/                  # Minimal echo environment template
├── possible-improvements/   # Research notes + papers per task area
└── docs/                    # Design plans, env analysis, known issues
```

## What a "good" agent looks like

| Metric | Cleaning | Enrichment | Answering |
|--------|----------|------------|-----------|
| Primary signal | DQ score > 0.95 | Coverage > 0.80 | Combined reward > 0.70 |
| Efficiency | < 15 steps | < 12 steps | 1 step |
| Format compliance | Valid JSON actions | Valid JSON actions | Answer + cited_columns + reasoning |
| Domain awareness | Correct columns for domain | Relevant sources for domain | Persona-appropriate language |
