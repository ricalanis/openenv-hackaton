# Enrichment Environment Improvements

## Current State

- Agent calls `lookup(domain, source, row)` from a fixed registry (5 sources per domain)
- No tradeoffs, no cost, no wrong answer — just "apply all enrichments"
- Coverage metric is essentially a checklist, not a decision problem
- Real benchmark: Both GPT-4o-mini and Qwen3-8B achieve 0.20 coverage (1/5 enrichments) per episode

## Proposed Improvements

### 1. Budget-Constrained Enrichment

**Problem:** No tradeoffs — apply all enrichments and win.
**Solution:** Limited enrichment budget forces prioritization.

```python
class EnrichmentEnvironment(Environment):
    def reset(self, seed=None, domain=None):
        # ...existing code...
        self._budget = 3  # can only apply 3 of 5 possible enrichments
        self._costs = {  # different enrichments cost different amounts
            "salary_band": 1,
            "tenure_risk": 1,
            "flight_risk_score": 2,  # composite = more expensive
            "industry_benchmark": 2,  # lookup = external cost
            "satisfaction_index": 1,
        }

    def step(self, action):
        source = action.source or action.field_name
        cost = self._costs.get(source, 1)

        if cost > self._budget:
            # Can't afford this enrichment
            return self._make_observation(error="Insufficient budget")

        self._budget -= cost
        # ...apply enrichment...
```

**Impact:** Agent must evaluate which enrichments provide the most value for the budget. Real decision-making under constraints.

### 2. Conflicting Sources

**Problem:** All enrichment sources are correct.
**Solution:** Some sources provide contradictory information; agent must choose.

```python
# Example: two salary benchmarks disagree
CONFLICTING_SOURCES = {
    "hr": {
        "industry_benchmark": {"Sales Executive": 65000, ...},
        "regional_benchmark": {"Sales Executive": 72000, ...},  # different!
    },
}

# Add confidence scores to sources
SOURCE_RELIABILITY = {
    "salary_band": 0.95,       # derived from data, very reliable
    "industry_benchmark": 0.80, # external, somewhat reliable
    "regional_benchmark": 0.70, # external, less reliable
}
```

**Impact:** Agent must reason about source reliability and resolve conflicts, not just apply everything.

### 3. Noisy Enrichment Sources

**Problem:** Enrichments are always correct.
**Solution:** Some enrichments have noise; agent must evaluate quality.

```python
def lookup_with_noise(domain, source, row, noise_rate=0.1):
    """Some enrichments return incorrect values."""
    true_value = lookup(domain, source, row)
    if random.random() < noise_rate:
        # Return a plausible but wrong value
        return generate_plausible_wrong_value(domain, source, true_value)
    return true_value
```

**Impact:** Agent learns to validate enrichment outputs, not blindly trust them.

### 4. Downstream-Aware Enrichment (Utility-Based Selection)

**Problem:** Coverage treats all enrichments equally.
**Solution:** Reward based on how much each enrichment helps the answering task.

```python
def enrichment_reward_v2(coverage, enrichment_utility):
    """Reward that considers downstream utility, not just coverage.

    enrichment_utility: how much this enrichment helps answering
    (e.g., from citation tracking or LLM-as-judge scoring)
    """
    return round(0.50 * coverage + 0.50 * enrichment_utility, 4)

# Track which enrichments get cited in answers
ENRICHMENT_CITATION_RATES = {
    "hr": {
        "flight_risk_score": 0.85,  # frequently cited in answers
        "salary_band": 0.75,
        "satisfaction_index": 0.60,
        "tenure_risk": 0.40,
        "industry_benchmark": 0.25,  # rarely cited
    },
}
```

**Impact:** Agent learns to prioritize enrichments that actually matter for business answers. Directly addresses mentor feedback on "propagating business value."

### 5. Time-Sensitive Enrichment

**Problem:** Enrichments are static.
**Solution:** Some enrichments deprecate or change over time steps.

```python
def step(self, action):
    # ...apply enrichment...

    # After each step, some existing enrichments may become stale
    for field in self._fields_added:
        if random.random() < 0.05:  # 5% chance per step
            self._stale_fields.add(field)

    # Stale fields don't count toward coverage
    active_coverage = len(self._fields_added - self._stale_fields) / len(possible)
```

**Impact:** Agent must consider temporal dynamics. Refreshing stale enrichments vs adding new ones.

### 6. Multi-Source Aggregation

**Problem:** Each enrichment comes from one source.
**Solution:** Some derived fields require combining multiple sources.

```python
COMPOSITE_ENRICHMENTS = {
    "hr": {
        "comprehensive_risk_score": {
            "requires": ["tenure_risk", "satisfaction_index"],
            "compute": lambda risks: 0.6 * risks["tenure_risk"] + 0.4 * (1 - risks["satisfaction_index"]),
            "description": "Combined risk from multiple factors",
        },
    },
}
```

**Impact:** Agent must plan enrichment order — some enrichments unlock more powerful composite enrichments.

## Recommended Implementation Order

1. Budget constraints (30 min, transforms from checklist to optimization)
2. Downstream-aware enrichment (1 hour, addresses mentor feedback)
3. Conflicting sources (1 hour, adds genuine decision-making)
4. Multi-source aggregation (2 hours, adds planning dimension)
5. Noisy sources + time-sensitivity (2 hours, adds realism)
