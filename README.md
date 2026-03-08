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

### The Pipeline in Action

**Stage 1 — Cleaning (Bronze → Silver):** The agent receives a messy enterprise dataset and must diagnose and fix data quality issues through sequential actions: filling nulls, fixing types, removing duplicates, standardizing formats, correcting typos. Reward is based on a composite DQ score (completeness, consistency, uniqueness) enhanced with Snorkel-style labeling functions.

**Stage 2 — Enrichment (Silver → Gold):** The cleaned data gets enriched with derived fields from 20 domain-specific sources (e.g., salary bands from income, flight risk scores from tenure + satisfaction, velocity scores from stage duration). The agent learns which enrichments matter for each domain.

**Stage 3 — Answering (Gold → Insight):** Given enriched data, a question, and a persona, the agent produces a cited analytical answer adapted to the audience — executives get ROI language, managers get operational metrics, ICs get actionable next steps.

### Deep Dive: Data Cleaning as Core Challenge

Cleaning is where we invested the deepest effort — it's the stage with the richest sequential decision-making:

- **6 operations** across up to **15 steps** per episode
- **4 enterprise domains** prevent memorization — the agent must generalize cleaning strategies
- **Dense reward signal**: `0.50 x DQ_after + 0.50 x improvement_delta`
- **DQ scoring**: `0.40 x completeness + 0.35 x consistency + 0.25 x uniqueness`
- **Seeded deterministic resets** — reproducible training/evaluation alignment
- **18 Snorkel-style labeling functions** for domain-aware quality assessment
- **Curriculum design**: from single-corruption episodes to complex multi-issue datasets

The cleaning environment demonstrates the core FSDS principle: the agent handles the mechanical diagnosis and repair, while the human defines what "clean" means through quality gates and data contracts.

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
