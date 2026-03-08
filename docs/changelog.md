# DataSage Changelog

## 2026-03-08 - Possible Improvements Research & Proposals

**New:** Created `possible-improvements/` folder with research-backed proposals for improving DataSage environments, training, and reward design. Organized by pipeline stage (cleaning, enrichment, answering) + cross-cutting concerns (pipeline reward propagation, GRPO algorithm improvements).

### Key Findings
- **Pipeline gap identified:** Original design doc specified `downstream_signal` (0.30 for cleaning, 0.50 for enrichment) but it was never implemented. Current rewards are purely proxy-based (DQ score, coverage), disconnected from business value.
- **Top improvement:** LLM-as-judge for downstream reward signal — evaluates whether cleaning/enrichment actions actually helped the final answer
- **GRPO fixes:** GDPO per-reward normalization, λ-GRPO for step-level fairness, MO-GRPO for auto-balanced reward weights
- **Environment innovations:** Budget-constrained enrichment, multi-turn answering, schema drift, dependent corruptions

### Files Created (11 total)
- `possible-improvements/README.md` — Overview and priority matrix
- `possible-improvements/{cleaning,enrichment,answering,pipeline,grpo-training}/README.md` — Proposals per area
- `possible-improvements/{cleaning,enrichment,answering,pipeline,grpo-training}/papers.md` — 30+ referenced papers

## 2026-03-08 - Academic Research Survey: RL Environment Improvements

**New:** Comprehensive literature survey (`docs/research/2026-03-08-rl-environment-improvement-papers.md`) covering 30+ papers across 8 research categories relevant to DataSage's RL training pipeline. Key findings: GDPO (NVIDIA, drop-in GRPO replacement for multi-reward normalization), turn-level credit assignment for multi-step cleaning episodes, VCRL variance-based curriculum learning, and MO-GRPO for automatic reward balancing. Priority recommendations organized by implementation effort vs. impact.

## 2026-03-08 - Inference Notebooks for GRPO Model Evaluation (Split for OOM)

**New:** Split inference into 5 standalone Colab-compatible notebooks (`demo/inference/`) to avoid OOM on T4 GPUs. Each notebook loads only one model at a time.

### Notebooks
| # | File | GPU | Description | Output |
|---|------|-----|-------------|--------|
| 0 | `00_wandb_export.ipynb` | No | Export W&B training metrics from runs `xuwyjpe6`, `orww3s2q`, `2mltqk5w` | `wandb_training_data.json` |
| 1 | `01_cleaning_inference.ipynb` | Yes | Cleaning LoRA + base model, 4 domains x 3 episodes | `cleaning_results.json` |
| 2 | `02_enrichment_inference.ipynb` | Yes | Enrichment LoRA + base model, 4 domains x 3 episodes | `enrichment_results.json` |
| 3 | `03_answering_inference.ipynb` | Yes | Answering LoRA + base model, 4 domains x 3 personas | `answering_results.json` |
| 4 | `04_aggregate_results.ipynb` | No | Combine all JSONs, comparison table, charts | `evaluation_results.json` |

### Memory management
- Each GPU notebook loads LoRA adapter → runs inference → `del model; torch.cuda.empty_cache()` → loads base model → runs comparison → frees again
- No two models in memory at the same time
- Output JSONs exclude raw action text to keep files small

*(Replaces single monolithic `datasage_inference.ipynb` which caused OOM)*

## 2026-03-08 - Multi-Model Demo with LangGraph Agentic System

**New:** Built a full interactive demo (`demo/`) with LangGraph agentic system comparing GPT-4o-mini, Qwen3-8B, and GRPO fine-tuned DataSage models. Ran **real benchmarks** against live HF Space environments.

### Components
- **LangGraph Agent** (`demo/backend/agent.py`): StateGraph with initialize → select_action → execute_action → evaluate loop, hitting live HF Space `/web/reset` and `/web/step` endpoints
- **Model Abstraction** (`demo/backend/models.py`): Pluggable providers - OpenAI (GPT-4o-mini) and HuggingFace Inference via fireworks-ai (Qwen3-8B), with Qwen3 `<think>` tag stripping
- **Real Benchmark Runner** (`demo/run_real_benchmarks.py`): Runs actual episodes against live environments, collects real metrics
- **Standard Benchmarks** (`demo/backend/standard_benchmarks.py`): MMLU, HumanEval, GSM8K, ARC-Challenge, HellaSwag, TruthfulQA, Winogrande + 8 domain-specific benchmarks
- **Gradio UI** (`demo/app.py`): 5-tab dashboard clearly labeling REAL vs PROJECTED results

### Real Benchmark Results (from live HF Space environments)
- **Answering** (clear differentiator): GPT-4o-mini: **0.712** vs Qwen3-8B: **0.515** mean reward (+38%)
- **Cleaning**: Both models ~0.96 reward (env starts above done threshold, 1-step episodes)
- **Enrichment**: Both models 0.20 coverage (1/5 enrichments per episode)
- DataSage fine-tuned models (LoRA adapters) require GPU inference, scores are projected

### Key Findings
- Answering task provides the strongest model differentiation
- Cleaning environment is trivially easy for any model (DQ starts at 0.96+)
- Enrichment equally hard for both - models struggle with multi-step enrichment chains
- DataSage GRPO training specifically targets these failure modes

## 2026-03-08 - Fix GRPO Completion Mask Misalignment (94 vs 42)

**Problem:** After fixing prompt length, GRPOTrainer crashed with `RuntimeError: tensor a (94) != tensor b (42)` in `masked_batch_mean`. Neither value matched `max_completion_length=256`.

**Root cause:** [Unsloth bug #3149](https://github.com/unslothai/unsloth/issues/3149) — GRPOTrainer ignores `gradient_accumulation_steps` when validating batch size divisibility by `num_generations`. With `batch_size=1, accumulation=8, num_generations=4`, Unsloth auto-adjusted batch_size to 4, making effective batch `4*8=32` instead of `1*8=8`. This misaligned which `completion_mask` was applied to which completion's values.

**Fix:** Set `gradient_accumulation_steps=1` and `num_generations=8` to match [Unsloth's official reference](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb). Updated all 3 notebooks, 3 .py scripts, and shared config.

## 2026-03-08 - Fix GRPO Completion/Mask Size Mismatch (338 vs 256)

**Problem:** After enabling `fast_inference`+`use_vllm`, GRPOTrainer crashed with `RuntimeError: The size of tensor a (338) must match the size of tensor b (256)` in `masked_batch_mean`. The 256 = `max_completion_length`, the 338 = actual completion tokens.

**Root cause:** Environment observations (DQ reports, data previews, column info) tokenize to 600-800+ tokens with the chat template. `max_prompt_length=512` was too small. With vLLM, the full untruncated prompt was sent to generation, producing completions whose length (when computed as `total - max_prompt_length`) exceeded `max_completion_length`.

**Fix:** Increased `MAX_PROMPT_LENGTH` from 512 to 1024 across all training files and shared config. Total `1024 + 256 = 1280` is well within `max_seq_length=2048`.

## 2026-03-08 - Fix GRPO Tensor Size Mismatch (254 vs 255)

**Problem:** GRPOTrainer crashed with `RuntimeError: The size of tensor a (254) must match the size of tensor b (255)` during training. Prior fixes (batch_size=1, fp16, dtype) didn't resolve it.

**Root cause:** Without `fast_inference=True` on `FastLanguageModel.from_pretrained()`, Unsloth uses HF's native `generate()` for GRPO's multi-generation step. Native generate doesn't properly pad/align sequences of different completion lengths when computing log probabilities. The earlier removal of `use_vllm` (commit 7979408) was correct for external vLLM, but Unsloth's `fast_inference=True` creates an internal vLLM engine that supports PEFT/LoRA.

**Fix (matches [Unsloth's official Qwen2.5-3B GRPO reference](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb)):**

### All 3 notebooks (cleaning, enrichment, answering)
- Added `fast_inference=True`, `max_lora_rank=16`, `gpu_memory_utilization=0.6` to `FastLanguageModel.from_pretrained()`
- Added `use_vllm=True` to `GRPOConfig` (uses Unsloth's internal vLLM, not external)
- Removed `dtype=torch.float16` (Unsloth handles dtype internally)
- Removed `fp16=True, bf16=False` from GRPOConfig
- Added optimizer settings from reference: `adam_beta1=0.9`, `adam_beta2=0.99`, `weight_decay=0.1`, `warmup_ratio=0.1`, `lr_scheduler_type="cosine"`, `optim="adamw_8bit"`, `max_grad_norm=0.1`

### All 3 .py scripts (cleaning, enrichment, answering)
- Added `fast_inference=True`, `max_lora_rank=16`, `gpu_memory_utilization=0.6` to `from_pretrained()` (already had `use_vllm=True`)

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
