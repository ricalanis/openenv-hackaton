# DataSage Changelog

## 2026-03-07 - Full Implementation (Phases 0-5)

### Phase 0: Foundation
- Created shared utilities: `environments/shared/` with 5 modules
  - `domains.py`: 4 enterprise domains (HR, Sales, PM, IT Ops) with DomainConfig
  - `enterprise_data.py`: data loading, corruption injection, DQ scoring
  - `reward_utils.py`: reward computation for all 3 stages
  - `personas.py`: 3 personas (Executive, Manager, IC) with alignment scoring
  - `enrichment_sources.py`: static lookup tables and derived computations per domain
- Created `data/labeling_functions.py`: 18 Snorkel-style LFs across 4 domains
- Created `data/fetch_datasets.py`: real data fetchers with synthetic fallbacks

### Phase 1: Environments (3 parallel)
- **Cleaning** (`environments/cleaning/`): 6 operations (fill_null, fix_type, remove_duplicate, standardize, trim, correct_typo), multi-step episodes, DQ reward
- **Enrichment** (`environments/enrichment/`): domain-specific enrichment sources, coverage tracking, done at 80%
- **Answering** (`environments/answering/`): persona-aware QA, faithfulness scoring, optional Patronus Lynx, single-step episodes
- All follow OpenEnv 0.2.1 scaffold (models, server, client, Dockerfile, openenv.yaml)

### Phase 2: Smoke Tests
- `tests/test_cleaning.py`: reset, step, all-domains verification
- `tests/test_enrichment.py`: reset, step, full-episode verification
- `tests/test_answering.py`: reset, good/bad answer, all-personas verification
- All tests pass across 4 domains and 3 personas

### Phase 4: Training Scripts
- `training/shared/config.py`: central config (model, URLs, hyperparams)
- `training/shared/parsers.py`: JSON extraction + keyword fallback for all 3 stages
- `training/train_cleaning.py`: Unsloth+GRPO, 64 multi-domain prompts, W&B
- `training/train_enrichment.py`: domain-aware enrichment, coverage reward
- `training/train_answering.py`: Patronus Lynx + persona reward

### Phase 5: Benchmarks
- `benchmarks/eval_cleaning.py`: DQ metrics per domain
- `benchmarks/eval_enrichment.py`: coverage ratio and information gain
- `benchmarks/eval_answering.py`: faithfulness + persona alignment + Patronus
- `benchmarks/persona_align.py`: PersonaAlignScorer
- `benchmarks/run_all.py`: orchestrator with W&B + E2E score
- Baseline E2E score: 0.5247

### Deployment
- `deployment/northflank/`: 4 H100 GPU job configs (3 stages + benchmarks)
