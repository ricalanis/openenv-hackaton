---
title: DataSage Cleaning Environment
emoji: 🧹
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# DataSage Cleaning Environment

An RL environment for training agents to clean enterprise data across four domains (HR, Sales, Project Management, IT Operations).

The agent receives a corrupted 50-row data batch and must apply cleaning operations (fill nulls, fix types, remove duplicates, standardize values, trim whitespace, correct typos) to maximise a composite data quality score. Episodes end when DQ > 0.95 or after 15 steps.

## Quick Start

```python
from environments.cleaning.models import CleaningAction
from environments.cleaning.client import CleaningEnv

with CleaningEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Domain: {result.observation.domain}, DQ: {result.observation.dq_score}")

    result = env.step(CleaningAction(
        operation="fill_null", column="Age", value="median"
    ))
    print(f"DQ after step: {result.observation.dq_score}")
```
