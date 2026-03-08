---
title: DataSage Answering Environment
emoji: "\U0001F4CA"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - datasage
---

# DataSage Answering Environment

A single-step RL environment where an agent answers enterprise data questions tailored to a specific persona (Executive, Manager, Individual Contributor) using enriched data context across 4 domains (HR, Sales, PM, IT Ops).

## Quick Start

```python
from environments.answering.models import AnsweringAction
from environments.answering.server.answering_environment import AnsweringEnvironment

env = AnsweringEnvironment()
obs = env.reset()
print(f"Domain: {obs.domain}, Persona: {obs.persona}")
print(f"Question: {obs.question}")

action = AnsweringAction(
    answer="Based on the data, key trends show...",
    cited_columns=obs.available_columns[:3],
    reasoning="Analyzed available columns for patterns."
)
result = env.step(action)
print(f"Reward: {result.reward}, Done: {result.done}")
```

## Reward

The reward combines:
- **Faithfulness** (0-1): Are cited columns valid? Does the answer reference real data values?
- **Persona relevance** (0-1): Does the answer match the persona's language style and focus areas?
- **Patronus score** (optional): If `PATRONUS_API_KEY` is set, uses Patronus Lynx for hallucination detection.

Without Patronus: `0.30 * faithfulness + 0.70 * persona_relevance`
With Patronus: `0.40 * patronus_score + 0.60 * persona_relevance`
