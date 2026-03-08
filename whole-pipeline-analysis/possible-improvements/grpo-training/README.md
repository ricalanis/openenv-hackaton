# GRPO Training Improvements

## Current State

- GRPO via Unsloth + TRL on Qwen 2.5-3B with LoRA (r=16)
- 3 epochs, batch_size=1, num_generations=8
- Multi-reward functions per stage (env_reward + format + auxiliary)
- Known issues: tensor mismatches, gradient_accumulation_steps=1 required

## Proposed Improvements

### 1. GDPO: Per-Reward Normalization (Drop-In Fix)

**Problem:** GRPO normalizes the combined reward, but when multiple reward functions have different scales/variances, one reward dominates and others are ignored.

**Solution:** GDPO (NVIDIA, 2026) normalizes each reward independently before aggregation.

```python
# Current approach (problematic):
# rewards = 0.7 * env_reward + 0.3 * format_reward
# advantage = (rewards - mean(rewards)) / std(rewards)  # combined normalization

# GDPO approach (fix):
def gdpo_normalize_rewards(reward_lists, weights):
    """Normalize each reward function independently, then combine."""
    normalized = []
    for rewards, weight in zip(reward_lists, weights):
        r = np.array(rewards)
        r_norm = (r - r.mean()) / (r.std() + 1e-8)
        normalized.append(weight * r_norm)
    return sum(normalized)
```

**Impact:** Prevents training signal collapse in multi-reward settings. All three DataSage stages use multiple rewards.
**Paper:** [GDPO: Learning to Directly Align Language Models with Diverse Feedback](https://arxiv.org/abs/2601.05242)
**Effort:** Very Low — modify reward computation in training loop.

### 2. λ-GRPO: Process-Step-Aware Scaling

**Problem:** GRPO assigns uniform advantage to all tokens in a completion. For multi-step episodes (cleaning: 15 steps, enrichment: 12 steps), early steps get the same credit as the crucial final steps.

**Solution:** λ-GRPO scales each process step's loss by 1/|λ| (the number of trajectories sharing the prefix), equalizing updates across steps.

```python
# Standard GRPO: all steps get same advantage
# λ-GRPO: per-step scaling
def lambda_grpo_loss(trajectories, advantages):
    """Scale each step's loss by inverse of prefix overlap count."""
    for traj in trajectories:
        for step_idx, step in enumerate(traj):
            # Count how many trajectories share this prefix
            lambda_count = count_shared_prefix(trajectories, traj, step_idx)
            step.loss_weight = 1.0 / lambda_count
```

**Impact:** Faster convergence, better exploration of step-level decisions.
**Paper:** [GRPO is Secretly a PRM](https://arxiv.org/abs/2509.21154) (Sullivan, 2025)
**Effort:** Medium — requires modifying the GRPO loss computation.

### 3. MO-GRPO: Auto-Balanced Multi-Objective

**Problem:** Reward weights are manually tuned (70/30, 30/70). Wrong weights waste training signal.

**Solution:** MO-GRPO auto-balances reward functions by their variance — high-variance rewards get lower weight to prevent instability.

```python
def auto_balance_rewards(reward_lists):
    """Auto-balance multiple reward functions by inverse variance."""
    weights = []
    for rewards in reward_lists:
        variance = np.var(rewards)
        weights.append(1.0 / (variance + 1e-8))
    # Normalize weights to sum to 1
    total = sum(weights)
    return [w / total for w in weights]
```

**Impact:** Eliminates manual tuning of the 70/30 splits in `reward_utils.py`.
**Paper:** [MO-GRPO](https://arxiv.org/abs/2509.22047) (2025)
**Effort:** Low — replace fixed weights with dynamic computation.

### 4. Turn-Level Credit Assignment

**Problem:** In multi-step episodes (cleaning: 15 steps), GRPO gives the entire trajectory one advantage estimate. The agent doesn't learn which specific step was good or bad.

**Solution:** Per-step advantage estimation using the environment's per-step rewards.

```python
def turn_level_advantages(trajectory, gamma=0.99):
    """Compute per-step advantages for multi-step episodes."""
    T = len(trajectory)
    returns = [0.0] * T

    # Compute discounted returns from the end
    returns[T-1] = trajectory[T-1].reward
    for t in range(T-2, -1, -1):
        returns[t] = trajectory[t].reward + gamma * returns[t+1]

    # Advantages = returns - baseline (mean return at that step)
    baseline = np.mean([returns[t] for t in range(T)])
    advantages = [returns[t] - baseline for t in range(T)]
    return advantages
```

**Impact:** Each cleaning/enrichment step gets its own training signal. Much faster learning on multi-step episodes.
**Paper:** [Turn-Level Credit Assignment for Multi-Turn RLHF](https://arxiv.org/abs/2505.11821) (Zeng et al., 2025)
**Effort:** Medium — requires modifying the rollout and advantage computation.

### 5. Curriculum Learning via Reward Variance (VCRL)

**Problem:** All training examples are sampled uniformly. Easy examples waste compute, hard examples produce noisy gradients.

**Solution:** Use reward variance across GRPO's N generations as a difficulty measure. Train on medium-difficulty examples.

```python
def select_curriculum_batch(dataset, model, num_generations=8):
    """Select training examples at optimal difficulty."""
    scored_examples = []
    for example in dataset:
        # Generate N completions
        completions = model.generate(example.prompt, n=num_generations)
        rewards = [evaluate(c) for c in completions]

        # Variance = difficulty proxy
        # Low variance = too easy (all good) or too hard (all bad)
        # High variance = right challenge level
        variance = np.var(rewards)
        scored_examples.append((example, variance))

    # Sort by variance, take top 50%
    scored_examples.sort(key=lambda x: x[1], reverse=True)
    return [ex for ex, _ in scored_examples[:len(scored_examples)//2]]
```

**Impact:** +24.8 points improvement reported in VCRL paper. Focuses training on examples where the model is actually learning.
**Paper:** [VCRL: Variance-Based Curriculum RL for LLMs](https://arxiv.org/abs/2509.19803) (Jiang et al., 2025)
**Effort:** Medium — requires pre-scoring dataset before each epoch.

### 6. Verifiable Binary Rewards (DeepSeek-R1 Style)

**Problem:** Continuous rewards (0.0-1.0) are noisy and hard to optimize.
**Solution:** Add binary pass/fail rewards alongside continuous scores.

```python
def verifiable_cleaning_reward(dq_after, format_valid):
    """Binary verifiable rewards + continuous signal."""
    binary_pass = 1.0 if dq_after > 0.95 else 0.0  # verifiable
    format_pass = 1.0 if format_valid else 0.0       # verifiable
    return 0.4 * dq_after + 0.3 * binary_pass + 0.3 * format_pass

def verifiable_enrichment_reward(coverage, all_valid_sources):
    binary_pass = 1.0 if coverage > 0.80 else 0.0
    source_valid = 1.0 if all_valid_sources else 0.0
    return 0.4 * coverage + 0.3 * binary_pass + 0.3 * source_valid
```

**Impact:** Binary rewards are proven to work for RL on LLMs (DeepSeek-R1). Clear, unambiguous learning signal.
**Paper:** [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (2025)
**Effort:** Very Low — add binary thresholds to existing rewards.

### 7. Trajectory Caching (Tool-R1)

**Problem:** Each GRPO generation is independent. Successful trajectories are discarded.
**Solution:** Cache high-quality trajectories and mix them into training batches.

```python
class TrajectoryCache:
    """Cache successful trajectories for reuse in training."""
    def __init__(self, max_size=1000):
        self.cache = []
        self.max_size = max_size

    def add(self, trajectory, reward):
        if reward > 0.8:  # only cache high-quality
            self.cache.append((trajectory, reward))
            if len(self.cache) > self.max_size:
                self.cache.pop(0)  # FIFO eviction

    def sample(self, n):
        return random.sample(self.cache, min(n, len(self.cache)))
```

**Impact:** Reuses successful episodes, especially valuable when environment calls are expensive (HF Space HTTP requests).
**Paper:** [Tool-R1](https://arxiv.org/abs/2509.12867) (Zhang et al., 2025)
**Effort:** Low — add cache alongside existing training loop.

## Recommended Implementation Order

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| 1 | GDPO per-reward normalization | 30 min | Fixes multi-reward training |
| 2 | Verifiable binary rewards | 30 min | Cleaner learning signal |
| 3 | MO-GRPO auto-balanced weights | 1h | Eliminates manual tuning |
| 4 | Trajectory caching | 1h | Reduces env call overhead |
| 5 | Turn-level credit assignment | 2h | Per-step learning for multi-step envs |
| 6 | Curriculum learning (VCRL) | 3h | Focuses training on useful examples |
| 7 | λ-GRPO step scaling | 3h | Faster convergence |
