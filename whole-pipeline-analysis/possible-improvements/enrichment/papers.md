# Relevant Papers — Enrichment Environment

## Information-Theoretic Approaches

### Information Bottleneck-Enhanced RL (2024)
- **Title:** Information Bottleneck-Enhanced Reinforcement Learning for State Representation
- **URL:** https://www.mdpi.com/1424-8220/25/24/7572
- **Key Idea:** Applies the Information Bottleneck principle to RL state representations. The optimal representation is minimal and sufficient for the downstream task — keep relevant info, discard noise.
- **DataSage Application:** Enrichment should maximize mutual information between enriched data and answer quality. Use I(enriched_data; answer_quality) as reward signal. Fields that are never cited in answers have zero mutual information.

### Multimodal Information Bottleneck for Deep RL (2024)
- **Title:** Multimodal Information Bottleneck for Deep Reinforcement Learning
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0893608024002715
- **Key Idea:** Measures information flow between processing stages. Optimizes for maximal relevant information transfer between stages.
- **DataSage Application:** Measure information flow from enrichment to answering. Which enriched fields carry the most bits of information relevant to answer quality?

## Multi-Agent / Multi-Stage Cooperation

### STARE-VLA: Stage-Aware Reinforcement (2025)
- **Title:** STARE-VLA: Progressive Stage-Aware Reinforcement for Robotics
- **URL:** https://arxiv.org/pdf/2512.05107
- **Key Idea:** Stage calculator computes per-stage costs and per-step rewards. Enables credit assignment among degrees of success at each stage, not just binary success/failure.
- **DataSage Application:** Compute per-enrichment utility scores based on how much each enrichment contributes to the answering stage's success.

### Explicit Credit Assignment through Local Rewards (2025)
- **Title:** Explicit Credit Assignment through Local Rewards and Dependence Graphs
- **URL:** https://arxiv.org/html/2601.21523v1
- **Key Idea:** When a team reward can be attributed to specific subsets of agents, local rewards provide more fine-grained learning signals than a shared global reward.
- **DataSage Application:** Decompose the pipeline E2E score into individual enrichment contributions. Each enrichment action gets its own local reward based on causal contribution to the answer.

## Resource-Constrained Decision Making

### Budget-Constrained Multi-Agent Systems (Survey)
- **Key Idea:** In real enterprise data enrichment, API calls cost money, lookups have latency, and compute is limited. Budget constraints transform simple coverage problems into knapsack-like optimization.
- **DataSage Application:** Adding a budget to enrichment transforms it from "apply everything" to a genuine optimization problem. The agent must learn enrichment value/cost ratios.

## Active Data Management

### Active Learning for Data Enrichment
- **Key Idea:** Active learning principles can guide which data points benefit most from enrichment. Not all rows need all enrichments — some rows are already information-rich.
- **DataSage Application:** Allow the agent to selectively enrich subsets of rows, not the entire DataFrame. Row-level enrichment decisions add a combinatorial explosion to the action space.
