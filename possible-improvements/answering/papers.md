# Relevant Papers — Answering Environment

## Multi-Turn Reasoning Agents

### Turn-Level Credit Assignment for Multi-Turn RLHF (2025)
- **Title:** Turn-Level Credit Assignment for Multi-Turn Reinforcement Learning from Human Feedback
- **Authors:** Zeng et al.
- **URL:** https://arxiv.org/abs/2505.11821
- **Key Idea:** Extends GRPO to a multi-turn variant with per-step advantage estimation. Solves the credit assignment problem where multi-step episodes get a single trajectory-level advantage.
- **DataSage Application:** Directly applicable if answering becomes multi-turn. Each turn (clarification, data request, draft answer, refinement) gets its own advantage estimate.

### Agent-R1: Training LLM Agents as Multi-Turn MDP (2025)
- **Title:** Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning
- **URL:** https://arxiv.org/abs/2511.14460
- **Key Idea:** Full MDP framework for multi-turn tool-calling agents. Formulates multi-step agent interactions as proper MDPs with per-step rewards and advantages.
- **DataSage Application:** Framework for converting the single-step answering env into a proper multi-turn MDP with tool calls (data queries, column lookups, persona adaptation).

### Tool-R1: Tool-Augmented RL for LLMs (2025)
- **Title:** Tool-R1: Reinforcement Learning for Tool-Augmented LLM Agents
- **Authors:** Zhang et al.
- **URL:** https://arxiv.org/abs/2509.12867
- **Key Idea:** Dynamic sample queue caching high-quality trajectories for reuse. Standardized tool-use API for code, SQL, and search.
- **DataSage Application:** The answering agent could have tools: `query_data(sql)`, `get_stats(column)`, `check_trend(column, period)`. Tool-use adds genuine multi-step reasoning.

### VerlTool: Unified Tool-Use RL Framework (2025)
- **Title:** VerlTool: Unified RL Framework for Tool-Using LLM Agents
- **URL:** https://arxiv.org/abs/2509.01055
- **Key Idea:** Standardized tool-use RL framework with verifiable execution environments. Provides APIs for code, SQL, and search tools within RL training loops.
- **DataSage Application:** Framework for adding data analysis tools to the answering environment.

## Persona-Aware Generation

### J1: RL-Trained LLM Judges (Meta, 2025)
- **Title:** J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning
- **URL:** https://arxiv.org/abs/2505.10320
- **Key Idea:** Uses RL to train LLM judges. Simple binary rewards for correct verdicts outperform complex multi-component schemes. Dynamic criteria generation and iterative self-correction.
- **DataSage Application:** Train a persona-specific judge model that evaluates whether answers match the persona's expectations. Replace keyword-based persona scoring with a learned judge.

### LLM-as-Judge for Reward Signals (2024)
- **Title:** LLM-as-a-Judge & Reward Model: What They Can and Cannot Do
- **URL:** https://arxiv.org/abs/2409.11239
- **Key Idea:** Comprehensive analysis of using LLMs as reward models. Shows strengths (nuanced evaluation) and weaknesses (position bias, verbosity bias).
- **DataSage Application:** Use LLM-as-judge for persona evaluation instead of keyword matching. More nuanced than the current `score_persona_alignment()` which just counts keyword hits.

## Verifiable Rewards

### DeepSeek-R1 (2025)
- **Title:** DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **URL:** https://arxiv.org/abs/2501.12948
- **Key Idea:** Binary verifiable rewards (accuracy + format) without any learned reward model. Shows that simple, verifiable rewards are sufficient for RL on LLMs.
- **DataSage Application:** Add verifiable binary rewards to answering: "Did the answer cite at least 2 valid columns?" (yes/no), "Is the answer > 100 chars?" (yes/no), "Does the answer contain a specific data value?" (yes/no).

### Patronus Lynx (2024)
- **Title:** Lynx: State-of-the-Art Open-Source Hallucination Detection Model
- **URL:** https://www.patronus.ai/blog/lynx-state-of-the-art-open-source-hallucination-detection-model
- **Key Idea:** Open-source hallucination detection model that provides verifiable faithfulness scores.
- **DataSage Application:** Already integrated as `patronus_score` in `answering_reward()`. Could be used more aggressively as a binary pass/fail gate.

## Adversarial Question Generation

### RICOL: Retrospective In-Context Learning (2025)
- **Title:** Retrospective In-Context Learning for Temporal Credit Assignment with LLMs
- **URL:** https://arxiv.org/html/2602.17497v1
- **Key Idea:** Uses LLMs to analyze past decisions and convert sparse rewards into dense signals. Leverages the LLM's pre-trained knowledge to estimate advantage functions with ~10 samples (vs ~1000 for Monte Carlo).
- **DataSage Application:** After a failed answering episode, use the LLM to analyze why the answer was poor and generate a better question that targets the same weakness. Self-evolving question bank.

### Latent Reward: LLM Credit Assignment (2024)
- **Title:** Latent Reward: LLM-Empowered Credit Assignment in Episodic Reinforcement Learning
- **URL:** https://arxiv.org/abs/2412.11120
- **Key Idea:** Uses LLMs for credit assignment in episodic RL. The LLM estimates which actions in a trajectory were most responsible for the final outcome.
- **DataSage Application:** For multi-turn answering, use the LLM to identify which turn (clarification, data request, answer draft) contributed most to the final quality.
