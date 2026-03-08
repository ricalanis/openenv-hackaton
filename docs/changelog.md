# DataSage Changelog

## 2026-03-08 - Make Training Notebooks Fully Self-Contained

**Problem:** Notebooks imported from the project repo (`training.shared.*`, `environments.*`), which broke in Colab because the repo clone + sys.path setup was fragile.

### All 3 notebooks (cleaning, enrichment, answering)
- Removed Cell 2 (repo clone + sys.path setup) entirely
- Removed all imports from `training.shared.*` and `environments.*`
- Removed `openenv-core` and `pydantic` from pip install
- Inlined config constants (`ENV_URL`, `BASE_MODEL`, hyperparams) directly in each notebook
- Inlined parser functions (`parse_cleaning_action`, `parse_enrichment_action`, `parse_answering_action`, `_extract_column`) as plain functions
- Inlined auxiliary reward functions (`cleaning_json_format_reward`, `source_relevance_reward`, `enrichment_json_format_reward`, `patronus_reward_fn`, `local_faithfulness_fn`, `answering_json_format_reward`, `persona_match_reward`)
- Inlined persona definitions (`Persona` dataclass, `PERSONAS`, `score_persona_alignment`, `_check_formality`) in answering notebook
- Replaced `EnvClient` usage with plain `requests.post()` to HF Space HTTP endpoints (`/reset`, `/step`)
- Only pip deps: `unsloth`, `trl`, `datasets`, `wandb`, `requests` (+ optional `patronus` for answering)

### Cell structure (all notebooks)
| # | Content |
|---|---------|
| 0 | Markdown header + Colab badge |
| 1 | `!pip install` (no openenv-core) |
| 2 | API keys (Colab Secrets) |
| 3 | Config + parser + auxiliary reward functions (all inlined) |
| 4 | Model loading (Unsloth) |
| 5 | System prompt + task descriptions |
| 6 | Dataset build (plain `requests` to HF Space) |
| 7 | Env reward function (plain `requests`) |
| 8 | GRPOConfig |
| 9 | Trainer + train |
| 10 | Save + push to Hub |

### Files unchanged
- Everything outside `training/train_*.ipynb` is unchanged
- Existing tests unaffected (they test server-side code, not notebooks)

## 2026-03-07 - Fix Colab Notebook Tensor Instability

**Problem:** Notebooks crashed with tensor length errors due to `rollout_func` + `generate_rollout_completions` manually managing `prompt_ids`/`completion_ids`/`logprobs` tensor lists. Any size mismatch crashed training.

### Notebooks (all 3: cleaning, enrichment, answering)
- Removed `rollout_func` pattern and `generate_rollout_completions` import
- Pre-build dataset with real env observations at creation time (64 examples via `reset_with_seed`)
- Env reward functions defined locally in each notebook — call env directly with stored seeds
- Seeds stored as dataset column, passed to reward functions via TRL kwargs
- Auxiliary reward functions (`source_relevance_reward`, `persona_match_reward`, etc.) receive context via dataset columns (`available_sources`, `persona_name`)
- Dropped `dq_improvement_reward` (redundant with env reward, required rollout kwargs)

### GRPOConfig alignment with OpenEnv tutorial
- `per_device_train_batch_size`: 4 → 2
- `num_generations`: 8 → 4
- `max_completion_length`: 512/768 → 256 (JSON actions are short)
- `max_prompt_length`: 1024 → 512
- Updated `training/shared/config.py` TRAINING_CONFIGS to match

### Files unchanged
- `training/shared/rewards.py` — auxiliary reward functions compatible as-is
- `training/shared/parsers.py` — reused unchanged
- `training/train_*.py` — standalone scripts not touched

## 2026-03-07 - Code Quality Refinements

### Reward Functions
- Extracted all training reward functions to `training/shared/rewards.py` (testable in isolation)
- Pre-compiled regex patterns (no duplicate `re.search` calls)
- `enrichment_reward` now has 0.2 completion bonus at 80% coverage threshold (was identity)
- Removed unused `torch` import from training scripts

### Training Notebooks
- Converted training scripts to standalone Colab notebooks (`training/train_*.ipynb`)
- Auto-detect Colab vs local environment, clone repo if needed
- Import reward functions from shared module instead of inline definitions

### API Validation
- Added Pydantic `ResetWithSeedRequest` model to all 3 `/reset-with-seed` endpoints
- Validates `seed: int | None` and `domain: str | None` types

### Tests
- Added 21 new tests for training reward functions (`tests/test_training_rewards.py`)
- Fixed `test_enrichment_seeded_reset_different_seeds` (had zero assertions)
- Total: 42 tests passing

## 2026-03-07 - Reward Pipeline Fix (rollout_func)

**Problem:** Training reward functions evaluated model completions against random environment state, not the context the model saw — rewards were noise. Persona reward checked all 3 personas instead of the requested one. `DOWNSTREAM_CACHE` added no information.

### Seeded Resets (all 3 environments)
- Added `seed` and `domain` params to `reset()` in Cleaning, Enrichment, and Answering environments
- Seeds `random.seed(seed)` and `np.random.seed(seed)` for deterministic state
- Cleaning: monkey-patches `np.random.default_rng` during corruption injection for full determinism
- Added `POST /reset-with-seed` endpoint to all 3 FastAPI apps (`app.py`)
- Added `reset_with_seed()` method to all 3 clients (`client.py`)

### Simplified `reward_utils.py`
- Removed `DOWNSTREAM_CACHE` and `_get_downstream_bucket` — redundant static lookup
- `cleaning_reward(dq_before, dq_after)` — now takes before/after for delta-based reward
- `enrichment_reward(coverage)` — direct signal, no downstream mixing
- `answering_reward(faithfulness, persona_relevance, patronus_score)` — clean weighted blend

### Training Script Rewrites
- **train_cleaning.py**: Rewritten with `rollout_func` + `generate_rollout_completions`
  - Replaced 64 static domain prompts with 8 generic task descriptions + live env observation injection
  - Model now sees real environment observations before generating actions
  - Rewards evaluate against the same seeded environment state (context-matched)
  - Added `dq_improvement_reward` (delta-based) replacing `reasoning_reward` (keyword-based)
  - Removed `reasoning_reward` (too shallow — keyword matching for "first", "let me", etc.)
- **train_enrichment.py**: Same `rollout_func` rewrite
  - Added `source_relevance_reward` — checks if model picked a valid source from available_sources
  - Removed `reasoning_reward`
- **train_answering.py**: Same `rollout_func` rewrite
  - **Fixed `persona_match_reward`**: reads requested persona from rollout kwargs, scores alignment against that specific persona only (was checking all 3 and taking max)
  - Kept `patronus_reward_fn` with local fallback

### Test Infrastructure
- Added `tests/conftest.py` to handle sys.path isolation (bare `from models import` caused cross-env conflicts when running all tests together)
- Added `tests/test_reward_utils.py` with 5 tests for simplified reward functions
- Added seeded reset tests to all 3 environment test files (`test_cleaning`, `test_enrichment`, `test_answering`)
- Total: 21 tests passing

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
