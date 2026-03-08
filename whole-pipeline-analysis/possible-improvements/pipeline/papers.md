# Relevant Papers — Pipeline-Level Reward Propagation

## End-to-End Differentiable Pipelines

### WindTunnel (VLDB 2022)
- **Title:** WindTunnel: Towards Differentiable ML Pipelines Beyond a Single Model
- **URL:** https://www.vldb.org/pvldb/vol15/p11-yu.pdf
- **Key Idea:** Translates trained ML pipeline operators (including non-differentiable ones) into differentiable neural network modules, then jointly fine-tunes the entire pipeline. Training operators in isolation is sub-optimal; joint training achieves higher accuracy.
- **DataSage Application:** The core architectural principle. DataSage's 3 stages train in isolation but should be optimized jointly. Even without literal differentiability, the insight applies: upstream decisions should be evaluated against downstream outcomes.

### DiffML / DiffPrep (2022-2023)
- **Title:** DiffML: End-to-end Differentiable ML Pipelines
- **URL:** https://arxiv.org/abs/2207.01269
- **Key Idea:** Makes data cleaning and feature engineering decisions differentiable so downstream task loss propagates to preprocessing choices. Uses Gumbel-Softmax for discrete decisions.
- **DataSage Application:** Framework for making cleaning operation selection differentiable. In practice, use REINFORCE-style gradient estimation since we can't literally backpropagate through the HF Space API.

### GRADE: Replace REINFORCE with Backprop for LLM Alignment (2025)
- **Title:** GRADE: Replacing Policy Gradients with Backpropagation
- **URL:** https://arxiv.org/html/2601.11574v1
- **Key Idea:** Uses Gumbel-Softmax to replace REINFORCE-style policy gradients with direct backpropagation through discrete sampling. Lower variance gradient estimation.
- **DataSage Application:** Could replace GRPO's policy gradient with lower-variance backprop-through-sampling for the discrete action spaces (operation selection, column selection).

## Hindsight and Relabeling

### Hindsight Experience Replay (NeurIPS 2017)
- **Title:** Hindsight Experience Replay
- **URL:** https://proceedings.neurips.cc/paper/7090-hindsight-experience-replay.pdf
- **Key Idea:** Converts failed episodes into synthetic successes by relabeling goals with what was actually achieved. Dramatically improves learning in sparse reward settings.
- **DataSage Application:** After full pipeline rollouts, relabel cleaning/enrichment rewards with the actual downstream answering outcome. Every trajectory teaches something, even "failed" ones.

### Relay HER: Sequential Task Decomposition (2023)
- **Title:** Relay Hindsight Experience Replay for Sequential Tasks
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0925231223007439
- **Key Idea:** Decomposes sequential tasks into subtasks with increasing complexity. Allows learning from the simplest subtask first, then gradually completing harder ones.
- **DataSage Application:** Maps directly to DataSage's staged pipeline. Start training cleaning first (simplest), then enrichment (medium), then answering (hardest), with downstream signal flowing back.

### Hindsight Task Relabelling for Meta-RL (NeurIPS 2021)
- **Title:** Hindsight Task Relabelling: Experience Replay for Sparse Reward Meta-RL
- **URL:** https://proceedings.neurips.cc/paper/2021/file/1454ca2270599546dfcd2a3700e4d2f1-Paper.pdf
- **Key Idea:** Generalizes HER to meta-RL settings. Relabels tasks (not just goals) in hindsight, enabling the agent to learn from attempts at different tasks.
- **DataSage Application:** Cross-domain transfer — relabel cleaning trajectories from one domain as training data for another domain.

## Credit Assignment and Value Decomposition

### QMIX: Monotonic Value Function Factorisation (2018)
- **Title:** QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL
- **URL:** https://arxiv.org/abs/1803.11485
- **Key Idea:** Decomposes a joint Q-value into individual agent Q-values with a monotonic mixing network. If any agent improves, the total improves. Enables decentralized execution with centralized training.
- **DataSage Application:** Treat the 3-stage pipeline as a cooperative multi-agent system. A QMIX-style mixer learns to combine stage-local rewards into a total pipeline reward while preserving the property that each stage is incentivized to improve.

### RICOL: LLM-Based Retrospective Credit Assignment (2025)
- **Title:** Retrospective In-Context Learning for Temporal Credit Assignment
- **URL:** https://arxiv.org/html/2602.17497v1
- **Key Idea:** Uses the LLM itself to analyze past decisions and convert sparse rewards into dense signals. ~10 samples vs ~1000 for Monte Carlo methods.
- **DataSage Application:** After a full pipeline episode, prompt the LLM: "Which cleaning actions were most important for the final answer quality?" Use the LLM's analysis as credit assignment.

## Reward Shaping

### Potential-Based Reward Shaping (Ng et al., 1999)
- **Title:** Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping
- **URL:** https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- **Key Idea:** Adding a potential-based shaping reward F(s,s') = γΦ(s') - Φ(s) preserves the optimal policy. Provably safe way to add dense rewards without changing what the agent learns to do.
- **DataSage Application:** Define Φ(data_state) as a function of data quality, enrichment count, and estimated answering readiness. Use Φ(s') - Φ(s) as a shaping bonus for cleaning and enrichment steps.

### HPRS: Hierarchical Potential-Based Reward Shaping (2024)
- **Title:** Hierarchical Potential-Based Reward Shaping from Task Specifications
- **URL:** https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1454188/full
- **Key Idea:** Extends PBRS to hierarchical task structures. Each level of the hierarchy gets its own potential function. Avoids delayed reward problems.
- **DataSage Application:** Three-level hierarchy: operation-level (within cleaning), stage-level (between stages), pipeline-level (E2E score). Each level has its own potential function.

### Dense Rewards from Language Model Critic (EMNLP 2024)
- **Title:** Dense Reward for Free in Reinforcement Learning from Human Feedback
- **URL:** https://aclanthology.org/2024.emnlp-main.515/
- **Key Idea:** Converts sparse episode-level rewards into per-token/per-span dense rewards using an LLM critic.
- **DataSage Application:** Transform the single-scalar-per-episode cleaning reward into per-operation feedback. Each cleaning action gets its own reward based on how much it improved the data state.

### Hierarchical Reward Models (2025)
- **Title:** HRM: Evaluating Reasoning at Both Fine-Grained and Coarse-Grained Levels
- **URL:** https://arxiv.org/abs/2503.13551
- **Key Idea:** Evaluates reasoning at operation-level AND stage-level simultaneously. Fine-grained rewards guide individual steps; coarse-grained rewards guide overall strategy.
- **DataSage Application:** Fine-grained: per-cleaning-action DQ delta. Coarse-grained: per-stage contribution to E2E score. Both signals combined in training.

### LGR2: Hindsight Reward Relabeling (2024)
- **Title:** LGR2: Language Guided Reward Relabeling for Accelerating RL
- **URL:** https://arxiv.org/abs/2406.05881
- **Key Idea:** Uses language-based hindsight to relabel rewards. Failed trajectories get relabeled with what they actually achieved, providing useful training signal.
- **DataSage Application:** Failed cleaning trajectories (didn't reach DQ > 0.95) get relabeled: "This trajectory achieved DQ 0.85, which is still useful for enrichment." Partial credit instead of zero reward.
