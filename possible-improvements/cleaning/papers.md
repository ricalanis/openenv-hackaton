# Relevant Papers — Cleaning Environment

## Data Quality as RL

### RLclean (2024)
- **Title:** RLclean: Reinforcement Learning for Automated Data Cleaning
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0020025524011952
- **Key Idea:** Integrates error detection and repair in one RL framework. The agent learns to compose cleaning operations in the right order. Schema-level and instance-level errors are handled jointly.
- **DataSage Application:** Validates DataSage's core premise. Their finding that operation ORDER matters (fix types before filling nulls, deduplicate before standardizing) could be encoded as a curriculum or reward bonus.

### Learn2Clean (2019)
- **Title:** Learn2Clean: Optimizing the Sequence of Tasks for Data Preprocessing
- **URL:** https://dl.acm.org/doi/10.14778/3352063.3352077
- **Key Idea:** Demonstrates that the order of preprocessing operations significantly impacts downstream ML task performance. Uses Q-learning to find optimal operation sequences.
- **DataSage Application:** Add operation-order awareness to the reward function. Reward sequences that follow learned optimal orderings.

### CleanSurvival (2025)
- **Title:** CleanSurvival: RL-based Data Preprocessing
- **URL:** https://arxiv.org/abs/2502.03946
- **Key Idea:** RL-based data preprocessing achieves optimal solutions in 9-14% of brute-force time. Shows that RL is computationally efficient for data cleaning optimization.
- **DataSage Application:** Confirms the efficiency argument for RL-based cleaning vs exhaustive search.

## Schema Drift & Distribution Shift

### DriftGuard (2026)
- **Title:** DriftGuard: Hierarchical Framework for Drift Detection and Remediation
- **URL:** https://arxiv.org/abs/2601.08928
- **Key Idea:** Five-module framework (statistical tests, column-level monitoring, schema comparison, semantic analysis, remediation planning) achieves 97.8% recall on drift detection.
- **DataSage Application:** Schema drift detection modules could be used to generate challenging cleaning scenarios. Agent must detect and handle drift before cleaning.

### Self-Healing ML (2024)
- **Title:** Self-Healing Machine Learning: A Framework for Autonomous Adaptation
- **URL:** https://arxiv.org/abs/2411.00186
- **Key Idea:** H-LLM uses LLMs to self-diagnose and self-adapt to distribution shifts without human intervention. Self-healing pipeline monitors, detects, and fixes data drift.
- **DataSage Application:** Agent could learn self-healing behaviors — detect schema drift, adapt cleaning strategy accordingly.

### Relational Data Cleaning Survey (2024)
- **Title:** Survey on Relational Data Cleaning Methods
- **URL:** https://link.springer.com/article/10.1007/s41019-024-00266-7
- **Key Idea:** Comprehensive taxonomy of schema-level (constraint violations, FD violations) vs instance-level (typos, missing values, outliers) errors. Covers 50+ cleaning methods.
- **DataSage Application:** Use taxonomy to design more diverse corruption types beyond the current 5 (nulls, type mismatches, duplicates, whitespace, typos). Add FD violations, cross-column constraints, outliers.

## Adversarial Environment Design

### PAIRED (2020)
- **Title:** PAIRED: Protagonist Antagonist Induced Regret Environment Design
- **URL:** https://arxiv.org/abs/2012.02096
- **Key Idea:** Three-player game: adversary generates environments, protagonist solves them, antagonist provides regret baseline. Produces challenging-but-learnable environments via minimax regret.
- **DataSage Application:** Train an adversarial corruption generator alongside the cleaning agent. The adversary learns to inject corruptions that are hard for the current cleaning model but still solvable.

### ADD: Adversarial Diffusion Design (2024)
- **Title:** Adversarial Diffusion Design for Environment Generation
- **URL:** https://arxiv.org/abs/2410.19715
- **Key Idea:** Uses regret-guided diffusion models to generate diverse, challenging environments. Better diversity than PAIRED-style approaches.
- **DataSage Application:** Use a small diffusion model to generate corruption patterns that maintain diversity while targeting model weaknesses.

## Curriculum Learning

### WebRL (ICLR 2025)
- **Title:** WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum
- **URL:** https://arxiv.org/abs/2411.02337
- **Key Idea:** Generates new training tasks from failed attempts. Self-evolving curriculum that adapts to the agent's current capabilities.
- **DataSage Application:** After failed cleaning episodes, generate new corrupted datasets targeting the same failure mode. Self-evolving difficulty.
