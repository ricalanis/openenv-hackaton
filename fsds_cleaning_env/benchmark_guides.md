After reading your project README, two benchmarks stand out as the **best fit for your agent**. I’ll explain why using the characteristics of your environment.

Your environment evaluates an agent that must:

* profile a dataset
* detect issues (duplicates, invalid tokens, schema problems)
* apply cleaning operations
* pass quality gates
* submit a cleaned table 

It is therefore **not just a coding benchmark**. It is an **interactive data-cleaning agent benchmark** with:

* tool use
* multi-step decision making
* environment feedback
* reward signals

So the benchmarks you choose should reflect **data-science workflows and agentic behavior**, not just code generation.

---

# Recommended Benchmark 1

# DS-1000

## Why it fits your project

Your environment requires the agent to perform **pandas-style cleaning and transformations**. 

DS-1000 contains many tasks that mirror those operations:

Typical operations tested:

* handling missing values
* joins / merges
* groupby aggregations
* reshaping tables
* feature engineering
* data type fixes

These are exactly the kinds of transformations your agent performs with:

```
apply_cleaning_operation(...)
```

in the environment.

### What DS-1000 measures well

It evaluates the **atomic data-science skills** needed by your agent:

* pandas fluency
* statistical transformations
* data manipulation correctness

### Metric

```
pass@1
```

### Target performance

| Level  | DS-1000 score |
| ------ | ------------- |
| weak   | <40%          |
| decent | ~50%          |
| strong | >60%          |
| SOTA   | ~70–75%       |

If your agent achieves **>60%**, its data-manipulation skills are competitive with top LLMs.

---

# Recommended Benchmark 2

# DA-Code (Data-Science Agent Benchmark)

## Why it fits your project

Your system is **not just a code generator**.

It requires:

* multi-step reasoning
* environment interaction
* tool use
* iterative improvement

Exactly like DA-Code.

DA-Code tasks look like this:

```
inspect dataset
clean columns
engineer features
train model
evaluate results
```

Your environment uses a similar pipeline:

```
profile_data
apply_cleaning_operation
run_quality_gates
submit_solution
```

So DA-Code measures the **agentic behavior** your project is targeting.

### What DA-Code evaluates

* planning
* iterative reasoning
* code execution
* data-analysis workflows

### Metric

```
task completion score
```

### Current SOTA

| Model              | Score   |
| ------------------ | ------- |
| GPT-4-class agents | ~30–35% |
| open models        | ~15–25% |

Even the best systems solve only about **1/3 of tasks**, so it’s a challenging benchmark.

---

# Why these two benchmarks together are ideal

They measure **two complementary capabilities** your project needs.

| Capability                    | Benchmark   |
| ----------------------------- | ----------- |
| data cleaning / pandas skills | **DS-1000** |
| agent workflow reasoning      | **DA-Code** |

Your environment tests both.

If your agent scores well on both, it strongly suggests:

* it understands data manipulation
* it can plan multi-step cleaning workflows

---

# How to adapt them to your project

I would evaluate your agent in three layers.

## 1 — Micro-skills (DS-1000)

Measure:

```
pandas correctness
data transformations
aggregation logic
```

---

## 2 — Agent capability (DA-Code)

Measure:

```
multi-step reasoning
tool usage
pipeline construction
```

---

## 3 — Your custom benchmark

Your environment already defines good metrics:

* success rate
* average return
* invalid actions
* steps per episode 

These are excellent **agent evaluation metrics**.

---

# Suggested evaluation stack for your project

Use this hierarchy:

```
Level 1
DS-1000

Level 2
DA-Code

Level 3
FSDSCleaningEnv evaluation set
```

Where level 3 measures **task-specific performance**.

---

# One more thing (important)

Your environment has a very strong design choice:

```
random dataset per episode
```

This prevents memorization and encourages generalization. 

Many research benchmarks **do not have this property**, which makes your environment particularly good for RL.

---

# If your goal is to publish or win a hackathon

I would report:

```
DS-1000 score
DA-Code score
FSDSCleaningEnv success rate
```

Together those three demonstrate:

* coding skill
* agent reasoning
* domain-specific cleaning ability

