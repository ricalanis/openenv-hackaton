# Cleaning Environment Improvements

## Current State

- 6 operations on a 50-row DataFrame (fill_null, fix_type, remove_duplicate, standardize, trim, correct_typo)
- Invalid operations silently ignored (no penalty)
- Dense reward from DQ score makes it relatively easy
- A rule-based system could achieve high scores: "find nulls → fill_null, find duplicates → remove_duplicate"
- Real benchmark: Both GPT-4o-mini and Qwen3-8B achieve ~0.96 DQ immediately

## Proposed Improvements

### 1. Step Cost (Efficiency Pressure)

**Problem:** Agent can try all operations without consequence.
**Solution:** Add a per-step penalty that forces the agent to be strategic.

```python
def cleaning_reward(dq_before, dq_after, step_number, max_steps):
    improvement = max(0.0, dq_after - dq_before)
    quality_signal = 0.50 * dq_after + 0.50 * min(improvement * 5.0, 1.0)
    step_penalty = 0.02 * step_number  # accumulates over episode
    return max(0.0, quality_signal - step_penalty)
```

**Impact:** Agent must learn to diagnose issues efficiently, not brute-force all operations.

### 2. Penalize Invalid Operations

**Problem:** Unknown ops and bad columns are silently ignored (no-ops).
**Solution:** Negative reward for wasted actions.

```python
# In step():
try:
    # apply operation...
    op_valid = True
except Exception:
    op_valid = False

if not op_valid:
    reward -= 0.1  # penalty for invalid action
```

**Impact:** Agent learns action preconditions (e.g., can't fix_type on categorical column).

### 3. Dependent Corruptions (Cascading Issues)

**Problem:** Corruptions are independent — fixing one doesn't reveal another.
**Solution:** After fixing a column, reveal hidden issues in dependent columns.

```python
# Example: fixing MonthlyIncome nulls reveals salary/dept mismatches
CORRUPTION_DEPENDENCIES = {
    "hr": {
        "MonthlyIncome": ["Department"],  # fixing income reveals dept inconsistencies
        "YearsAtCompany": ["PerformanceRating"],  # fixing tenure reveals rating issues
    },
    "sales": {
        "Amount": ["Stage", "Probability"],  # fixing amount reveals stage mismatches
    },
}

def _inject_dependent_corruption(self, fixed_column):
    deps = CORRUPTION_DEPENDENCIES.get(self._domain_name, {})
    if fixed_column in deps:
        for dep_col in deps[fixed_column]:
            # Inject new corruption in dependent column
            self._inject_single_corruption(dep_col)
```

**Impact:** Agent must plan multi-step strategies. Requires tree-search-like reasoning.

### 4. Schema Drift (Patronus AI Prize Alignment)

**Problem:** Schema is fixed across episodes — no adaptation required.
**Solution:** Schema changes between episodes: columns renamed, types changed, new unknown columns appear.

```python
SCHEMA_DRIFT_SCENARIOS = [
    {"rename": {"MonthlyIncome": "monthly_income"}, "type_change": {"Age": "string"}},
    {"add_column": "LastReviewDate", "drop_column": "Education"},
    {"merge_columns": ("FirstName", "LastName", "FullName")},
]

def reset(self, seed=None, domain=None, schema_drift_level=0):
    # Normal reset...
    if schema_drift_level > 0:
        drift = random.choice(SCHEMA_DRIFT_SCENARIOS[:schema_drift_level])
        self._df = self._apply_schema_drift(self._df, drift)
        # Agent must detect and handle the drift
```

**Impact:** Agent can't memorize column names. Must reason about data semantics. Directly targets Patronus AI "Consumer Workflows with Schema Drift" prize.

### 5. Adversarial Corruption (Harder-to-Detect Issues)

**Problem:** Corruptions are obvious (nulls, "N/A" strings, exact duplicates).
**Solution:** Subtle corruptions that require reasoning to detect.

```python
SUBTLE_CORRUPTIONS = {
    "hr": [
        # Plausible but wrong: salary too low for role
        ("MonthlyIncome", "role_salary_mismatch", lambda df:
            df.loc[df['JobRole'] == 'Manager', 'MonthlyIncome'].apply(
                lambda x: max(1000, x * 0.3) if random.random() < 0.2 else x)),
        # Semantic duplicate: same person, slightly different name
        ("EmployeeID", "soft_duplicate", lambda df:
            inject_near_duplicate_rows(df, fuzz_columns=["Department"])),
    ],
}
```

**Impact:** Agent must understand domain semantics, not just pattern-match null/type checks.

### 6. Downstream-Aware Cleaning

**Problem:** DQ score treats all columns equally, but some columns matter more for answering.
**Solution:** Weight DQ score by column importance for downstream tasks.

```python
# Columns that appear in answering questions more often get higher weight
COLUMN_IMPORTANCE = {
    "hr": {"MonthlyIncome": 2.0, "Attrition": 2.0, "Department": 1.5,
            "Education": 0.5, "DistanceFromHome": 0.3},
    "sales": {"Amount": 2.0, "Stage": 2.0, "Probability": 1.5,
              "LeadSource": 0.5},
}

def compute_weighted_dq_score(df, domain_config, domain_name):
    weights = COLUMN_IMPORTANCE.get(domain_name, {})
    # Same DQ computation but weighted by column importance
```

**Impact:** Agent learns to prioritize cleaning the columns that matter for business answers.

## Recommended Implementation Order

1. Step cost + invalid op penalty (30 min, immediate difficulty increase)
2. Downstream-aware cleaning (1 hour, addresses mentor feedback)
3. Dependent corruptions (2 hours, makes env genuinely novel)
4. Schema drift (3 hours, targets Patronus AI prize)
5. Adversarial corruption (2 hours, adds depth)
