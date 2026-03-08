---
title: DataSage Enrichment Environment
emoji: "\U0001F4CA"
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# DataSage Enrichment Environment

An RL environment for enterprise data enrichment. The agent receives a cleaned
dataset from one of four enterprise domains (HR, Sales, PM, IT Ops) and must
enrich it by adding derived fields, lookups, and computed categories from the
domain's enrichment registry.

## Domains

- **HR & People**: salary bands, tenure risk, flight risk scores
- **Sales & Revenue**: deal size categories, velocity scores, win probability
- **Project Management**: schedule risk, resource utilization, burndown rates
- **IT Operations**: SLA compliance, MTTR bands, severity scores

## Quick Start

```python
from environments.enrichment import EnrichmentEnv
from environments.enrichment.models import EnrichmentAction

with EnrichmentEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Domain: {result.observation.domain}")
    print(f"Sources: {result.observation.available_sources}")

    result = env.step(EnrichmentAction(
        operation="add_field",
        field_name="salary_band",
        source="salary_band",
    ))
    print(f"Coverage: {result.observation.enrichment_coverage}")
```

## Reward

`0.50 * coverage + 0.50 * downstream_signal`

Done when coverage > 0.80 or step_count >= 12.
