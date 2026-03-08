# Pipeline-Level Improvements: Propagating Business Value

This is the **highest-priority improvement area**, directly addressing mentor feedback: "propagate business value to the layers before."

## The Problem

DataSage trains three stages independently. Each optimizes its own proxy metric:

| Stage | Current Reward | What it optimizes for |
|-------|---------------|----------------------|
| Cleaning | `0.50 * dq_after + 0.50 * improvement` | Data quality score |
| Enrichment | `coverage` | Number of enrichments applied |
| Answering | `0.30 * faithfulness + 0.70 * persona` | Answer quality (business value) |

The business value (answer quality) only exists in Stage 3. Stages 1 and 2 optimize for proxies that may not correlate with actual downstream utility. A cleaning action that improves DQ score but removes useful information for answering is rewarded. An enrichment that increases coverage but is never cited in answers is rewarded equally.

## The Original Design Intent

The design doc (`docs/plans/2026-03-07-datasage-design.md`) originally specified:
- Cleaning: `0.70 * dq_score + 0.30 * downstream_signal`
- Enrichment: `0.50 * coverage + 0.50 * downstream_signal`

This was never implemented. The fix is to add `downstream_signal` back.

## Proposed Approaches (Ordered by Practicality)

### 1. LLM-as-Judge Downstream Signal (Fastest to Implement)

After a full pipeline episode (Clean → Enrich → Answer), use an LLM judge to evaluate each stage's contribution.

```python
JUDGE_PROMPT = """You are evaluating a data pipeline.

## Original Dirty Data
{dirty_data_preview}

## Cleaning Actions Taken
{cleaning_actions}

## Data After Cleaning
{cleaned_data_preview}

## Enrichment Actions Taken
{enrichment_actions}

## Business Question
Persona: {persona} | Question: {question}

## Final Answer
{answer}

Rate each stage's contribution (0.0-1.0):
1. CLEANING_SCORE: Did cleaning fix issues that would cause bad answers?
2. ENRICHMENT_SCORE: Did enrichments add information used in the answer?
3. ANSWER_QUALITY: Is the answer faithful and persona-appropriate?

Respond as JSON: {{"cleaning_score": float, "enrichment_score": float, "answer_quality": float}}
"""

# Integration with reward_utils.py:
def cleaning_reward(dq_before, dq_after, downstream_signal=None):
    improvement = max(0.0, dq_after - dq_before)
    local = 0.50 * dq_after + 0.50 * min(improvement * 5.0, 1.0)
    if downstream_signal is not None:
        return round(0.70 * local + 0.30 * downstream_signal, 4)
    return round(local, 4)

def enrichment_reward(coverage, downstream_signal=None):
    if downstream_signal is not None:
        return round(0.50 * coverage + 0.50 * downstream_signal, 4)
    return round(coverage, 4)
```

**Cost:** One additional LLM call per full pipeline episode.
**Effort:** ~1 hour to implement (modify `reward_utils.py` + add judge call in rollout).

### 2. Hindsight Experience Replay (HER) for Reward Relabeling

After collecting full pipeline rollouts, retroactively relabel upstream rewards based on the final answering outcome.

```python
def hindsight_relabel(cleaning_trajectories, final_answer_rewards):
    """HER-style: use actual downstream outcome to relabel upstream rewards."""
    relabeled = []
    for traj, answer_reward in zip(cleaning_trajectories, final_answer_rewards):
        for step in traj:
            original_r = step['reward']
            # Blend original reward with downstream outcome
            step['hindsight_reward'] = 0.70 * original_r + 0.30 * answer_reward
        relabeled.append(traj)
    return relabeled
```

**Training workflow:**
1. Run full pipeline: Clean → Enrich → Answer (collect all rewards)
2. Relabel upstream rewards with downstream outcome
3. Use relabeled trajectories for GRPO training of upstream stages
4. Repeat

**Effort:** ~2 hours (modify training loop to collect and relabel trajectories).

### 3. Citation-Based Enrichment Reward

Track which enriched fields get cited in answers. Use citation frequency as a proxy for downstream utility.

```python
# After collecting many pipeline episodes:
CITATION_RATES = {}  # field_name -> fraction of answers that cite it

def enrichment_utility_reward(field_added, citation_rates):
    """Reward enrichments proportional to how often they're used downstream."""
    base_rate = 0.1
    return citation_rates.get(field_added, base_rate)
```

**Effort:** ~1 hour (add citation tracking to answering env, compute rates offline).

### 4. Potential-Based Reward Shaping (Theoretically Sound)

Define a potential function over data states. The shaping reward `F(s,s') = γ·Φ(s') - Φ(s)` provably preserves the optimal policy (Ng et al., 1999).

```python
def data_potential(null_rate, type_consistency, enrichment_count, max_enrichments):
    """Potential function: increases as data becomes more answering-ready."""
    completeness = 1.0 - null_rate
    enrichment_ratio = enrichment_count / max(1, max_enrichments)
    return 0.4 * completeness + 0.3 * type_consistency + 0.3 * enrichment_ratio

def shaped_reward(base_reward, state_before, state_after, gamma=0.99):
    phi_before = data_potential(**state_before)
    phi_after = data_potential(**state_after)
    shaping = gamma * phi_after - phi_before
    return base_reward + 0.3 * shaping
```

**Training protocol:**
1. Collect 100+ full pipeline rollouts
2. Fit regression: `answer_quality = f(cleaning_features, enrichment_features)`
3. Use predictions as potential function
4. Add shaping bonus to upstream rewards
5. Retrain with shaped rewards

**Effort:** ~3 hours (requires pipeline data collection + regression fitting).

### 5. Frozen-Model Pipeline Evaluation (Gold Standard)

After each cleaning episode, run the cleaned data through frozen enrichment + answering models to estimate downstream quality.

```python
def cleaning_reward_with_pipeline_eval(
    dq_before, dq_after, cleaned_df, domain, frozen_enrichment_model, frozen_answering_model
):
    """Run full pipeline evaluation after cleaning."""
    # Frozen enrichment
    enriched_df = frozen_enrichment_model.enrich(cleaned_df, domain)
    # Frozen answering
    answer_quality = frozen_answering_model.answer(enriched_df, domain, random_persona())

    local = cleaning_reward(dq_before, dq_after)
    return 0.70 * local + 0.30 * answer_quality
```

**Effort:** ~4 hours (requires frozen model inference during training).
**Cost:** Significant compute — doubles inference cost per training episode.

### 6. QMIX-Style Reward Mixer (Advanced)

Train a small neural network to combine stage rewards, respecting monotonicity constraints.

```python
class PipelineRewardMixer(nn.Module):
    """QMIX-inspired: monotonic mixing ensures each stage is incentivized to improve."""
    def forward(self, stage_rewards, state):
        # Hypernetwork generates mixing weights from state
        w = torch.abs(self.hyper_w(state))  # abs ensures monotonicity
        return (stage_rewards * w).sum(dim=-1)
```

**Effort:** ~5 hours (requires collected pipeline data + small model training).

## Recommended Implementation Order

| Priority | Approach | Effort | Impact |
|----------|----------|--------|--------|
| 1 | LLM-as-Judge downstream signal | 1h | Directly fills the design doc gap |
| 2 | Citation-based enrichment reward | 1h | Concrete, measurable signal |
| 3 | Hindsight relabeling | 2h | Leverages existing rollout infrastructure |
| 4 | Potential-based shaping | 3h | Theoretically grounded |
| 5 | Frozen-model evaluation | 4h | Gold standard but compute-heavy |
| 6 | QMIX mixer | 5h | Impressive but complex |
