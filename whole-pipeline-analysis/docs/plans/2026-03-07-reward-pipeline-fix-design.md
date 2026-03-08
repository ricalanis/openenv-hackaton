# DataSage Reward Pipeline Fix Design

**Date:** 2026-03-07
**Status:** Implemented
**Problem:** Training reward functions evaluate model completions against random environment state, not the context the model saw. Rewards are noisy and don't reflect action quality.

## Root Causes

1. **Context mismatch**: `env_reward_fn` calls `reset()` (random domain/corruption) then `step()` for each completion. The model generates based on a static prompt but the reward comes from unrelated env state.
2. **Redundant downstream signal**: `DOWNSTREAM_CACHE` is a static lookup of the score being mixed with — adds no information.
3. **Persona reward doesn't check requested persona**: Counts keyword hits for all 3 personas, rewards the max.
4. **Shallow auxiliary rewards**: Keyword matching teaches style, not substance.

## Solution: `rollout_func` with `generate_rollout_completions`

### Approach

Use TRL's `rollout_func` parameter on GRPOTrainer. This gives full control over the generation loop while remaining compatible with vLLM colocate mode.

### Architecture

```
For each prompt in batch:
  1. Call env.reset(seed=S, domain=D) via HTTP API
  2. Build full_prompt = system_prompt + env_observation + task_description
  3. Call generate_rollout_completions(trainer, [full_prompt] * num_gens)
  4. For each of N completions:
     a. Parse completion into action
     b. Call env.reset(seed=S, domain=D) again (same state)
     c. Call env.step(action) -> get reward, dq_score, etc.
  5. Return {prompt_ids, completion_ids, logprobs, env_reward, dq_before, dq_after, persona, ...}
```

### HTTP API Changes

Add `POST /reset-with-seed` endpoint to each environment server:

```json
{
  "seed": 42,
  "domain": "hr"  // optional, defaults to random
}
```

This wraps `random.seed(seed)` + `np.random.seed(seed)` around the existing reset logic, ensuring identical state for all N generations of the same prompt.

### Reward Function Changes

#### Cleaning

- **env_reward**: `dq_after` from rollout kwargs (raw score, no downstream mixing)
- **dq_improvement_reward** (NEW): `max(0, dq_after - dq_before) * 5.0` — rewards the delta
- **json_format_reward**: Kept as-is (valid JSON with correct schema)

#### Enrichment

- **env_reward**: `coverage` from rollout kwargs (raw score)
- **source_relevance_reward** (NEW): Checks if chosen source matches the task prompt's request
- **json_format_reward**: Kept as-is

#### Answering

- **env_reward**: `faithfulness + persona_alignment` from env (correctly matched to prompt context)
- **patronus_reward_fn**: Kept (with local fallback)
- **json_format_reward**: Kept as-is
- **persona_match_reward** (FIXED): Reads the requested persona from rollout kwargs, checks alignment with that specific persona only

#### Removed

- `DOWNSTREAM_CACHE` and all downstream signal mixing in `reward_utils.py`
- `reasoning_reward` (too shallow — keyword matching for "first", "let me", etc.)

### Updated `reward_utils.py`

```python
def cleaning_reward(dq_before: float, dq_after: float) -> float:
    improvement = max(0, dq_after - dq_before)
    return 0.50 * dq_after + 0.50 * min(improvement * 5.0, 1.0)

def enrichment_reward(coverage: float) -> float:
    return coverage  # direct signal, no downstream mixing

def answering_reward(faithfulness: float, persona_relevance: float,
                     patronus_score: float | None = None) -> float:
    if patronus_score is not None:
        return 0.40 * patronus_score + 0.60 * persona_relevance
    return 0.30 * faithfulness + 0.70 * persona_relevance
```

### Files to Modify

1. `environments/cleaning/server/cleaning_environment.py` — add seeded reset
2. `environments/enrichment/server/enrichment_environment.py` — add seeded reset
3. `environments/answering/server/answering_environment.py` — add seeded reset
4. `environments/cleaning/server/app.py` — expose /reset-with-seed endpoint
5. `environments/enrichment/server/app.py` — expose /reset-with-seed endpoint
6. `environments/answering/server/app.py` — expose /reset-with-seed endpoint
7. `environments/shared/reward_utils.py` — simplify, remove downstream cache
8. `training/train_cleaning.py` — replace dataset + reward_funcs with rollout_func
9. `training/train_enrichment.py` — same
10. `training/train_answering.py` — same
11. `environments/cleaning/client.py` — add reset_with_seed method
12. `environments/enrichment/client.py` — add reset_with_seed method
13. `environments/answering/client.py` — add reset_with_seed method

### Dependencies

- TRL with `rollout_func` support (experimental, available in recent versions)
- `trl.experimental.openenv.generate_rollout_completions`
- No changes to Unsloth or vLLM configuration

### Risks

- `rollout_func` is experimental — API may change
- HTTP latency for seeded resets (N+1 calls per prompt instead of 1)
- `generate_rollout_completions` must handle `num_generations` manually
