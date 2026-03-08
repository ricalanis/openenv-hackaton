# Answering Environment Improvements

## Current State

- Single-step episode: one question, one answer, done
- Not really RL — it's supervised evaluation wrapped in an env interface
- No sequential decision-making or adaptation
- This is acknowledged in `docs/environment_analysis.md` as the weakest environment
- Real benchmark: GPT-4o-mini scores 0.71, Qwen3-8B scores 0.52

## Proposed Improvements

### 1. Multi-Turn Dialogue (Transform to True RL)

**Problem:** Single-step means no sequential decision-making.
**Solution:** Multi-turn conversation where the agent can ask clarifying questions, get feedback, and refine.

```python
class MultiTurnAnsweringEnvironment(Environment):
    def __init__(self):
        self._max_turns = 5
        self._conversation_history = []
        self._feedback_model = None  # or rule-based feedback

    def reset(self, seed=None, domain=None):
        # Same initial setup: domain, persona, question, data
        self._conversation_history = []
        self._turn = 0
        return AnsweringObservation(
            # ...same fields...
            done=False,  # NOT done after first turn
            turn_number=0,
            max_turns=self._max_turns,
        )

    def step(self, action):
        self._turn += 1
        self._conversation_history.append(action.answer)

        if action.action_type == "ask_clarification":
            # Agent asks a clarifying question → gets feedback
            feedback = self._generate_feedback(action.answer)
            return AnsweringObservation(
                feedback=feedback,
                done=False,
                reward=0.0,  # no reward for asking
                turn_number=self._turn,
            )

        elif action.action_type == "submit_answer":
            # Agent submits final answer → gets terminal reward
            reward = self._evaluate_answer(action)
            # Bonus for fewer turns (efficiency)
            efficiency_bonus = 0.1 * (self._max_turns - self._turn) / self._max_turns
            return AnsweringObservation(
                done=True,
                reward=reward + efficiency_bonus,
            )

        elif action.action_type == "request_more_data":
            # Agent can request additional data columns/stats
            extra_stats = self._provide_extra_stats(action.requested_columns)
            return AnsweringObservation(
                column_stats=extra_stats,
                done=False,
                reward=0.0,
            )

        # Auto-end if max turns reached
        if self._turn >= self._max_turns:
            reward = self._evaluate_answer(action)
            return AnsweringObservation(done=True, reward=reward * 0.8)  # penalty for using all turns

    def _generate_feedback(self, draft_answer):
        """Persona gives feedback on draft answer."""
        feedbacks = {
            "Executive": [
                "I need ROI numbers, not just trends.",
                "How does this compare to last quarter?",
                "What's the bottom-line impact?",
            ],
            "Manager": [
                "What are the actionable next steps?",
                "Which team members are affected?",
                "What's the timeline for this?",
            ],
            "Individual Contributor": [
                "Can you explain this more simply?",
                "What should I do first?",
                "How does this affect my tasks?",
            ],
        }
        return random.choice(feedbacks.get(self._persona.name, ["Can you elaborate?"]))
```

**Impact:** Transforms answering from supervised evaluation to genuine multi-step RL. Agent learns when to ask for clarification, when to request more data, when to submit. Massive improvement in environment innovation score.

### 2. Adversarial/Tricky Questions

**Problem:** Questions are straightforward and answerable.
**Solution:** Add questions designed to test agent boundaries.

```python
ADVERSARIAL_QUESTIONS = {
    "hr": [
        # Requires reasoning about missing data
        "What's the average salary of employees who left? Note: some salary data is missing.",
        # Requires hedging (not enough data)
        "Can you predict which employees will leave next quarter?",
        # Trick question (correlation ≠ causation)
        "Does overtime cause attrition?",
        # Cross-column reasoning
        "Are highly educated employees more satisfied? Consider confounders.",
    ],
    "sales": [
        # Requires stating uncertainty
        "Will we hit our quarterly target?",
        # Requires multi-column analysis
        "What's driving the slowdown in the pipeline?",
    ],
}
```

**Impact:** Agent must learn to hedge, state uncertainty, and reason about data limitations — not just regurgitate statistics.

### 3. Iterative Refinement with Scoring Feedback

**Problem:** No feedback loop — agent submits once and it's over.
**Solution:** Agent sees its score and can try again (with diminishing returns).

```python
def step(self, action):
    self._turn += 1
    score = self._evaluate_answer(action)

    if score > 0.9 or self._turn >= self._max_turns:
        return AnsweringObservation(done=True, reward=score)

    # Provide partial feedback
    feedback = self._generate_scoring_feedback(action, score)
    # E.g., "Your faithfulness score was low. The answer didn't cite specific data values."
    #        "The persona alignment was weak. An executive wants ROI language."

    return AnsweringObservation(
        done=False,
        reward=score * 0.3,  # partial reward
        feedback=feedback,
        turn_number=self._turn,
    )
```

**Impact:** Agent learns self-correction behavior. Early answers can be exploratory, refined based on feedback.

### 4. Cross-Domain Transfer Questions

**Problem:** Each question is about one domain.
**Solution:** Questions that require combining insights from multiple domains.

```python
CROSS_DOMAIN_QUESTIONS = [
    {
        "question": "How does employee satisfaction (HR) correlate with project delivery rates (PM)?",
        "domains_needed": ["hr", "pm"],
        "requires_join": True,
    },
    {
        "question": "Are the teams with highest IT incident rates also the ones with lowest sales performance?",
        "domains_needed": ["it_ops", "sales"],
    },
]
```

**Impact:** Agent must synthesize information across domain boundaries.

### 5. Dynamic Persona Switching

**Problem:** Persona is fixed for the entire episode.
**Solution:** Persona changes mid-conversation (e.g., manager joins a call that started with an executive).

```python
def step(self, action):
    self._turn += 1

    # 20% chance persona switches after turn 2
    if self._turn > 2 and random.random() < 0.2:
        old_persona = self._persona
        self._persona = random.choice([p for p in PERSONAS if p != old_persona])
        return AnsweringObservation(
            persona=self._persona.name,
            persona_description=self._build_persona_desc(),
            feedback=f"[{self._persona.name} has joined the conversation. Please adapt your response.]",
            done=False,
        )
```

**Impact:** Agent must adapt communication style dynamically, not just match a fixed template.

## Recommended Implementation Order

1. Multi-turn dialogue (2 hours, biggest impact on env innovation score)
2. Iterative refinement with scoring feedback (1 hour, builds on multi-turn)
3. Adversarial questions (1 hour, adds depth)
4. Dynamic persona switching (30 min if multi-turn already exists)
5. Cross-domain transfer (2 hours, complex but impressive)
