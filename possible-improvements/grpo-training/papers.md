# Relevant Papers — GRPO Training Improvements

## GRPO Algorithm Improvements

### GDPO: Fixing GRPO for Multi-Reward Settings (NVIDIA, 2026)
- **Title:** GDPO: Learning to Directly Align Language Models with Diverse Feedback
- **URL:** https://arxiv.org/abs/2601.05242
- **Key Idea:** Drop-in replacement for GRPO in TRL that normalizes each reward independently before aggregation, preventing training signal collapse in multi-reward settings. When rewards have different scales/variances, combined normalization causes one reward to dominate.
- **DataSage Application:** All three DataSage stages use multiple reward functions. GDPO is a minimal-effort, high-impact change. Replace the single advantage normalization with per-reward normalization.

### λ-GRPO: Process-Step-Aware Scaling (2025)
- **Title:** GRPO is Secretly a PRM
- **Authors:** Sullivan et al.
- **URL:** https://arxiv.org/abs/2509.21154
- **Key Idea:** GRPO induces an implicit process reward model. Non-uniformly distributed process steps inject a group-size bias. λ-GRPO scales each step's loss by 1/|λ| (inverse prefix overlap), equalizing updates. Achieves higher validation accuracy and faster convergence.
- **DataSage Application:** For multi-step cleaning (15 steps) and enrichment (12 steps), λ-GRPO ensures each step gets fair credit. Prevents early steps from being over/under-weighted due to prefix sharing.

### MO-GRPO: Multi-Objective Auto-Balancing (2025)
- **Title:** MO-GRPO: Multi-Objective Group Relative Policy Optimization
- **URL:** https://arxiv.org/abs/2509.22047
- **Key Idea:** Auto-balances multiple reward functions by variance. High-variance rewards get lower weight. Eliminates manual tuning of reward weights.
- **DataSage Application:** Replace the fixed 70/30 and 30/70 splits in `reward_utils.py` with dynamic variance-based balancing. The system automatically finds the right balance between DQ score and format reward, between faithfulness and persona alignment.

### Success Amplification Proves GRPO Monotonicity (2025)
- **Title:** Success Amplification: Provable Guarantees for GRPO
- **Authors:** Mroueh et al.
- **URL:** https://arxiv.org/abs/2503.06639
- **Key Idea:** Proves that GRPO with binary rewards monotonically amplifies success probability. Provides theoretical foundation for using binary verifiable rewards.
- **DataSage Application:** Justifies adding binary pass/fail rewards (DQ > 0.95, valid JSON, coverage > 0.80) alongside continuous scores. The theoretical guarantee means these rewards provably improve the model.

## Curriculum Learning

### VCRL: Variance-Based Curriculum for GRPO (2025)
- **Title:** VCRL: Variance-Based Curriculum Reinforcement Learning for LLMs
- **Authors:** Jiang et al.
- **URL:** https://arxiv.org/abs/2509.19803
- **Key Idea:** Uses reward variance within GRPO's N generations as a zero-parameter difficulty measure. High variance = medium difficulty = optimal learning signal. Low variance = too easy or too hard. Achieves +24.8 points improvement.
- **DataSage Application:** For cleaning, high variance in DQ rewards across 8 generations identifies the "Goldilocks zone" batches. Auto-selects training data at optimal difficulty without manual curriculum design.

### AdaRFT: Adaptive Curriculum for RL Fine-Tuning (2025)
- **Title:** AdaRFT: Adaptive Reinforcement Fine-Tuning with Curriculum Learning
- **Authors:** Shi et al.
- **URL:** https://arxiv.org/abs/2504.05520
- **Key Idea:** 2x training speedup through adaptive curriculum. Dynamically adjusts problem difficulty based on current model capability.
- **DataSage Application:** Start with easy corruptions (few nulls), progress to hard ones (schema drift + dependent corruptions) as the model improves.

### Actor-Curator: Neural Curator for Problem Selection (2026)
- **Title:** Actor-Curator: Learned Problem Selection for RL Training
- **URL:** https://arxiv.org/abs/2602.20532
- **Key Idea:** Learns a neural curator that selects training problems by directly optimizing expected policy improvement. 80% speedup over uniform sampling.
- **DataSage Application:** Learn which domain/corruption combinations provide the most training signal for each stage.

### WebRL: Self-Evolving Online Curriculum (ICLR 2025)
- **Title:** WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum
- **URL:** https://arxiv.org/abs/2411.02337
- **Key Idea:** Generates new training tasks from failed attempts. The curriculum evolves to target the agent's current weaknesses.
- **DataSage Application:** After failed cleaning/enrichment episodes, generate new corrupted datasets targeting the same failure mode. Self-evolving difficulty.

## Multi-Step Agent Training

### Turn-Level Credit Assignment (2025)
- **Title:** Turn-Level Credit Assignment for Multi-Turn RLHF
- **Authors:** Zeng et al.
- **URL:** https://arxiv.org/abs/2505.11821
- **Key Idea:** Extends GRPO to multi-turn with per-step advantage estimation. Each turn in a multi-turn episode gets its own advantage, not a shared trajectory-level one.
- **DataSage Application:** For 15-step cleaning and 12-step enrichment episodes, each step gets its own advantage. The agent learns "step 3 was the crucial one" rather than "the whole episode was good."

### Agent-R1: Multi-Turn MDP for LLM Agents (2025)
- **Title:** Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning
- **URL:** https://arxiv.org/abs/2511.14460
- **Key Idea:** Full MDP framework for multi-turn tool-calling agents. Standardized training loop for agents that interact with environments over multiple steps.
- **DataSage Application:** Framework for training the cleaning and enrichment agents as proper multi-turn MDPs with per-step rewards and discounted returns.

### Tool-R1: Dynamic Sample Queue (2025)
- **Title:** Tool-R1: Reinforcement Learning for Tool-Augmented LLM Agents
- **Authors:** Zhang et al.
- **URL:** https://arxiv.org/abs/2509.12867
- **Key Idea:** Dynamic sample queue caching high-quality trajectories for reuse. Reduces the number of expensive environment interactions needed.
- **DataSage Application:** Cache successful cleaning/enrichment trajectories. Especially valuable when environment calls go through HF Space HTTP endpoints (slow).

## Verifiable Rewards

### DeepSeek-R1 (2025)
- **Title:** DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **URL:** https://arxiv.org/abs/2501.12948
- **Key Idea:** Binary verifiable rewards (accuracy + format) without any learned reward model. Shows that simple, verifiable rewards are sufficient for training strong reasoning models.
- **DataSage Application:** DQ > 0.95 is a natural binary verifiable reward. JSON format validity is another. Adding binary rewards alongside continuous scores gives clearer learning signal.

### PRM Lessons: Monte Carlo Estimation is Unreliable (2025)
- **Title:** Lessons from Scaling Process Reward Models
- **URL:** https://arxiv.org/abs/2501.07301
- **Key Idea:** Monte Carlo estimation for process reward labeling is unreliable. Rule-based verifiable rewards are more robust than learned reward models for intermediate steps.
- **DataSage Application:** Use rule-based verifiable rewards for cleaning (DQ threshold) and enrichment (coverage threshold) rather than trying to learn a process reward model. The environment itself provides verifiable ground truth.

### Patronus Lynx: Verifiable Faithfulness (2024)
- **Title:** Lynx: State-of-the-Art Open-Source Hallucination Detection Model
- **URL:** https://www.patronus.ai/blog/lynx-state-of-the-art-open-source-hallucination-detection-model
- **Key Idea:** Open-source model for verifiable hallucination detection. Provides binary pass/fail faithfulness evaluation.
- **DataSage Application:** Already partially integrated (`patronus_score` in `answering_reward()`). Could be used as a binary verifiable reward gate: answer passes faithfulness check or doesn't.

## Reward Shaping

### Comprehensive Overview of Reward Engineering (2024)
- **Title:** Comprehensive Overview of Reward Engineering and Shaping for RL
- **URL:** https://arxiv.org/html/2408.10215v1
- **Key Idea:** Survey of reward engineering techniques for RL. Covers potential-based shaping, intrinsic motivation, curiosity-driven exploration, and multi-objective reward design.
- **DataSage Application:** Reference for choosing the right reward shaping approach for each stage. Potential-based shaping for cleaning (dense per-step), intrinsic motivation for enrichment (curiosity about data), multi-objective for answering (faithfulness + persona + format).

### GRPO++ Tricks (2025)
- **Title:** GRPO++: Tricks for Making RL Actually Work for LLMs
- **URL:** https://cameronrwolfe.substack.com/p/grpo-tricks
- **Key Idea:** Practical tricks for GRPO training: reward clipping, KL penalty tuning, generation diversity, batch size selection. Empirical findings from practitioners.
- **DataSage Application:** Direct practical guidance for the existing GRPO training loop. Check that KL penalty, reward clipping, and generation temperature are set correctly.
