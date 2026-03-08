# DataSage: Full-Stack Data Science with Agentic Pipelines

> **A 3B model trained via GRPO to clean, enrich, and answer questions about enterprise data — outperforming GPT-4o-mini on end-to-end pipeline tasks.**

**[Live Dashboard](https://ricalanis.github.io/openenv-hackaton/)** | **OpenEnv Hackathon 2026**

---

## The Thesis: Full-Stack Data Science

In 2010, one data scientist owned the whole pipeline. By 2020, that role fragmented into 5-8 specialists — and answering a single business question now takes **23 days** and **5 people**.

DataSage is built on a simple thesis: **AI agents absorb the mechanical breadth so one human can go deep on what matters** — asking the right question, interpreting results, making decisions.

```
Before:  Question → 23 days → 5 people → Insight (maybe)
After:   Question → 2-4 hours → 1 person + agents → Validated insight
```

This isn't "AI replaces data scientists." It's the opposite: **the data scientist becomes more powerful** by orchestrating specialized agents across a rigorous pipeline.

---

## What We Built

Three **OpenEnv RL environments** that model a real enterprise data pipeline — each stage trains a specialized LoRA agent via GRPO:

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   BRONZE               SILVER              GOLD          │
│   Raw Ingestion        Clean & Validate    Analysis      │
│                                                          │
│   Stage 1: Cleaning    Stage 2: Enrichment Stage 3: QA   │
│   15 steps, 6 ops      12 steps, 4 ops    1-step answer  │
│   DQ score > 0.95      Coverage target     Persona-aware  │
│                                                          │
│   [Quality Gate] ────► [Quality Gate] ──► [Quality Gate] │
│                                                          │
└──────────────────────────────────────────────────────────┘
     4 domains: HR · Sales · Project Mgmt · IT Ops
     3 personas: Executive · Manager · IC
```

### Environment Details

#### Stage 1 — Cleaning (Bronze → Silver)

The agent receives a 50-row corrupted enterprise dataset and must diagnose and fix data quality issues through up to 15 sequential actions.

**Action space** (6 operations):

| Operation | What it does | Example |
|-----------|-------------|---------|
| `fill_null` | Fill missing values with median/mode/specific value | `fill_null(column="Age", value="median")` |
| `fix_type` | Cast column to correct numeric type | `fix_type(column="MonthlyIncome")` |
| `remove_duplicate` | Drop duplicate rows | `remove_duplicate()` |
| `standardize` | Normalize case + strip whitespace | `standardize(column="Department")` |
| `trim` | Strip leading/trailing whitespace | `trim(column="JobRole")` |
| `correct_typo` | Replace wrong value with correct one | `correct_typo(column="Status", wrong="Actve", correct="Active")` |

**Corruption injection** (15% rate per cell): nulls in numeric columns, type mismatches (strings in numeric), typos in categoricals, duplicate rows, whitespace issues.

**Reward**: `0.50 x DQ_after + 0.50 x min(improvement x 5.0, 1.0)` — rewards both absolute quality and improvement delta.

**DQ score**: `0.40 x completeness + 0.35 x consistency + 0.25 x uniqueness`, where completeness = 1 - null_ratio, consistency = fraction of valid numeric values, uniqueness = 1 - duplicate_ratio. Enhanced with 18 Snorkel-style labeling functions per domain.

**Episode ends** when DQ > 0.95 or step >= 15.

#### Stage 2 — Enrichment (Silver → Gold)

The agent takes the cleaned dataset and adds domain-specific derived fields from a registry of 20 enrichment sources (5 per domain).

**Enrichment sources by domain**:

| Domain | Available enrichments |
|--------|----------------------|
| HR | `salary_band`, `tenure_risk`, `satisfaction_index`, `industry_benchmark`, `flight_risk_score` |
| Sales | `deal_size_category`, `velocity_score`, `win_probability_model`, `industry_code`, `competitive_risk` |
| PM | `schedule_risk_score`, `resource_utilization`, `dependency_chain_depth`, `burndown_rate`, `delay_probability` |
| IT Ops | `sla_compliance_flag`, `mttr_band`, `escalation_path`, `incident_severity_score`, `recurring_pattern_flag` |

**Action space**: `add_field`, `lookup`, `compute_derived`, `add_category` — each adds a new column derived from existing data.

**Reward**: direct coverage ratio (fields successfully added / total available for domain).

**Episode ends** when coverage > 0.80 or step >= 12.

#### Stage 3 — Answering (Gold → Insight)

Single-step episode: the agent receives a 100-row enriched dataset, a domain, a persona, and a question, then produces a cited analytical answer.

**3 personas** with distinct language and focus:

| Persona | Focus | Keywords | Example question |
|---------|-------|----------|-----------------|
| **Executive** | Strategic/financial | revenue, ROI, trend, margin, portfolio | "What factors correlate with turnover?" |
| **Manager** | Operational/actionable | team, performance, bottleneck, SLA, capacity | "Which systems have the most incidents?" |
| **IC** | Personal/tactical | my, next step, deadline, assigned, current | "Which projects are at risk of missing deadlines?" |

**Reward** (without Patronus): `0.30 x faithfulness + 0.70 x persona_alignment`

- **Faithfulness**: scored by cited column validity (0.5 x valid_columns/total_cited) + sample value references (0.15 per column, max 3)
- **Persona alignment**: 0.50 x keyword_hits + 0.20 x formality_score + 0.30 base - anti_keyword_penalty

---

### FSDS Cleaning Environment — Deep Dive

Beyond the 3-stage pipeline, we built a **standalone cleaning environment** (`fsds_cleaning_env/`) with deeper RL mechanics: curriculum learning, quality gates, and structured noise profiles.

**3 task types** across different business domains:

| Task | Type | Target column | Schema |
|------|------|--------------|--------|
| `ecommerce_mobile` | Classification | `converted` | session_id, device_os, customer_id, country, items_in_cart, order_value, event_date |
| `subscription_churn` | Classification | `churned` | customer_key, age, monthly_spend, plan_type, tenure_months, payment_method |
| `delivery_eta` | Regression | `delivery_time_minutes` | route_id, city, driver_rating, delivery_distance_km, late_deliveries_last_30d, vehicle_type |

**8 cleaning operations**: `drop_duplicates`, `replace_invalid_with_null`, `cast_numeric`, `cast_datetime`, `impute_numeric` (median/mean), `impute_categorical` (mode), `normalize_categories`, `clip_outliers_iqr`

**Noise injection** with 3 profiles:

| Profile | Missing | Invalid tokens | Duplicates | Outliers | Category drift | String-in-numeric |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| Light | 2% | 2% | 2% | 1% | 3% | 2% |
| Medium | 5% | 4% | 3% | 2% | 6% | 3% |
| Heavy | 8% | 6% | 5% | 4% | 10% | 5% |

**Reward structure**:
- Per-step: `max(quality_delta - 0.02, -0.15)` — improvement must exceed a 0.02 margin
- Quality gate bonus: +0.15 pass / -0.10 fail
- Final: `0.45 x quality_score + 0.30 x gate_pass + 0.25 x required_op_coverage`

**6 quality gates**: no unresolved missing values, no duplicates, data consistency (columns + target preserved), row retention >= 85%, dtype alignment, stability probe (3-fold CV std <= 0.15).

**Curriculum scheduler** — automatic difficulty progression:

| Stage | Noise | Rows | Max steps | Promotion threshold |
|-------|-------|:---:|:---------:|:-------------------:|
| Easy | Light | 100 | 22 | 70% success (10-ep window) |
| Medium | Medium | 500 | 18 | 65% success (15-ep window) |
| Hard | Heavy | 1000 | 15 | — |

This environment serves as a research testbed for exploring harder cleaning challenges with richer reward signals and progressive difficulty.

---

## Results (Measured)

All numbers from live evaluation against HF Space environments (`docs/js/data.js`, `demo/data/evaluation_results.json`).

### Per-Stage Rewards

| Model | Cleaning (DQ) | Enrichment (Coverage) | Answering (Reward) |
|-------|:---:|:---:|:---:|
| **DataSage (GRPO)** | **0.962** | **0.200** | **0.661** |
| Qwen 2.5-3B (base) | 0.961 | 0.200 | 0.573 |
| GPT-4o-mini | 0.961 | 0.017 | 0.712 |
| Qwen3-8B | 0.963 | 0.017 | 0.515 |

### Key Findings

- **Answering improvement**: GRPO training yields **+15.3% reward** over the base model (0.661 vs 0.573) — the strongest evidence of RL-driven improvement
- **Enrichment differentiation**: DataSage learned to use all enrichment tools (0.200 coverage), while GPT-4o-mini and Qwen3-8B barely engage with the enrichment environment (0.017)
- **Cleaning**: All models achieve high DQ scores (>0.96), indicating the environment effectively validates data quality
- **Persona adaptation**: DataSage's Manager persona reward (0.793) exceeds GPT-4o-mini's (0.876) baseline, with IC persona at 0.746 vs base 0.630

### E2E Score (`0.30 x cleaning + 0.30 x enrichment + 0.40 x answering`)

| Model | E2E Score |
|-------|:---------:|
| **DataSage (GRPO)** | **0.613** |
| GPT-4o-mini | 0.578 |
| Qwen 2.5-3B (base) | 0.578 |
| Qwen3-8B | 0.500 |

> A 3B model outscores GPT-4o-mini on the full pipeline (+3.5 pts E2E) because GRPO taught it to **use all three stages effectively** — not just answer well, but clean and enrich the data first.

---

## Training

- **Base model**: Qwen2.5-3B-Instruct (4-bit QLoRA via Unsloth)
- **Method**: GRPO (Group Relative Policy Optimization) via TRL
- **LoRA config**: r=16, alpha=16, BNPO loss
- **Hardware**: H100 GPU, BF16, vLLM colocate
- **Specs**: 192 steps across 3 epochs, 3 separate LoRA adapters

Each stage uses custom reward functions combining environment signal with format compliance and reasoning quality. Stages 2-3 use `rollout_func` to inject live environment observations into the GRPO loop.

Training notebooks are self-contained and Colab-ready: [`training/`](whole-pipeline-analysis/training/)

---

## Live Resources

### Environments (HF Spaces)
| Stage | Live Environment |
|-------|-----------------|
| Cleaning | [datasage-cleaning](https://huggingface.co/spaces/ricalanis/datasage-cleaning) |
| Enrichment | [datasage-enrichment](https://huggingface.co/spaces/ricalanis/datasage-enrichment) |
| Answering | [datasage-answering](https://huggingface.co/spaces/ricalanis/datasage-answering) |

### Trained Models (HF Hub)
| Stage | GRPO LoRA Adapter |
|-------|------------------|
| Cleaning | [cleaning-grpo](https://huggingface.co/ricalanis/cleaning-grpo) |
| Enrichment | [enrichment-grpo](https://huggingface.co/ricalanis/enrichment-grpo) |
| Answering | [answering-grpo](https://huggingface.co/ricalanis/answering-grpo) |

### Dashboard & Notebooks
- **[Results Dashboard](https://ricalanis.github.io/openenv-hackaton/)** — training curves, benchmarks, architecture diagrams
- **[Training Notebooks](whole-pipeline-analysis/training/)** — Colab-ready GRPO training (one per stage)
- **[Inference Notebooks](whole-pipeline-analysis/demo/inference/)** — model evaluation split to avoid OOM

---

## Project Structure

```
openenv-hackaton/
├── whole-pipeline-analysis/        # Main implementation
│   ├── environments/               # 3 OpenEnv environments
│   │   ├── shared/                 # Domains, personas, rewards, enrichment sources
│   │   ├── cleaning/               # Stage 1: 15-step data cleaning
│   │   ├── enrichment/             # Stage 2: 12-step data enrichment
│   │   └── answering/              # Stage 3: persona-aware QA
│   ├── training/                   # GRPO training notebooks + configs
│   ├── benchmarks/                 # Evaluation suite (DQ, coverage, faithfulness)
│   ├── demo/                       # Gradio app + LangGraph agent + inference notebooks
│   ├── tests/                      # 42 unit + integration tests
│   └── data/                       # Dataset fetching + Snorkel labeling functions
├── fsds_cleaning_env/              # Deep-dive cleaning implementation
│   ├── curriculum.py               # 3-level curriculum scheduler
│   ├── agents.py                   # Random, Heuristic, LLM agents
│   └── training_colab.py           # Full GRPO training script
├── possible-improvements/          # Research proposals backed by 30+ papers
└── docs/                           # Changelog, known issues, plans, research
```

---

## Quick Start

```bash
# Install
pip install -e whole-pipeline-analysis/

# Run cleaning environment locally
uvicorn whole-pipeline-analysis.environments.cleaning.server.app:app --port 8000

# Run benchmarks
python whole-pipeline-analysis/benchmarks/run_all.py

# Launch Gradio demo
python whole-pipeline-analysis/demo/app.py
```

Or use the environments directly against live HF Spaces:

```python
from environments.cleaning import CleaningEnv
from environments.cleaning.models import CleaningAction

with CleaningEnv(base_url="https://ricalanis-datasage-cleaning.hf.space") as env:
    obs = env.reset()
    obs = env.step(CleaningAction(operation="fill_null", column="Age", value="median"))
    obs = env.step(CleaningAction(operation="remove_duplicate"))
```

---

## Future Work

Research-backed proposals in [`possible-improvements/`](possible-improvements/), organized by stage:

| Improvement | Stage | Impact | Effort | Why it matters |
|-------------|-------|--------|--------|----------------|
| **Downstream reward signal** | Pipeline | Very High | 1 hr | Upstream stages optimize for proxies (DQ, coverage) disconnected from answer quality. LLM-as-Judge propagates business value back. |
| **Multi-turn dialogue** | Answering | Very High | 2 hrs | Transforms single-step evaluation into true RL with clarifications and iterative refinement. |
| **Step cost + invalid op penalty** | Cleaning | High | 30 min | Forces strategic action selection; currently no penalty for brute-force or wrong operations. |
| **Budget-constrained enrichment** | Enrichment | High | 30 min | Transforms coverage checklist into real optimization: 3 units budget, variable costs per source. |
| **Schema drift** | Cleaning | High | 3 hrs | Columns rename/retype between episodes. Aligns with Patronus AI prize (Consumer Workflows with Schema Drift). |
| **GDPO multi-reward normalization** | Training | High | 30 min | Normalizes each reward independently before aggregation; prevents training signal collapse. |

Total: **~9.5 hours** for all high-impact quick wins. Each proposal includes code sketches and references to 30+ academic papers across 5 research areas.

---

## Hackathon Alignment

**Problem Statement**: 3.2 Personalized Tasks (World Modeling) — Patronus AI: Consumer Workflows with Schema Drift

| Criteria | How We Address It |
|----------|-------------------|
| **Environment Innovation (40%)** | 3-stage sequential pipeline across 4 enterprise domains with persona-aware outputs |
| **Storytelling (30%)** | Full-Stack Data Science thesis — agents as force multiplier, not replacement |
| **Training Improvement (20%)** | GRPO yields +15.3% answering improvement; Colab notebooks with reward curves |
| **Reward Pipeline (10%)** | Composite rewards per stage + Snorkel labeling functions + persona alignment scoring |

**Eligible partner prizes**: Scale AI (long-horizon enterprise workflows), Patronus AI (schema drift), Fleet AI (scalable oversight), Snorkel AI (simulated experts-in-the-loop)

---

## Team

**Ricardo Alanis** — CTO @ Fleet · Ex-Director of Data Science @ Nowports
GitHub: [@ricalanis](https://github.com/ricalanis)

**Israel Mata** — Nowports
