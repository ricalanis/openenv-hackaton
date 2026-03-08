"""
DataSage Answering Environment Implementation.

A single-step RL environment where the agent must answer enterprise data
questions tailored to a specific persona (Executive, Manager, IC) using
enriched data context across 4 domains (HR, Sales, PM, IT Ops).
"""

import os
import random
import sys
from uuid import uuid4

import numpy as np
import pandas as pd

# Load .env file for PATRONUS_API_KEY and other secrets
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'))
except ImportError:
    pass

# Allow imports from the project root so shared modules are reachable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from environments.shared.domains import DOMAINS
from environments.shared.enterprise_data import load_domain_data, format_preview
from environments.shared.personas import PERSONAS, score_persona_alignment
from environments.shared.reward_utils import answering_reward

from models import AnsweringAction, AnsweringObservation
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State


class AnsweringEnvironment(Environment):
    """
    Answering environment: the agent receives a data context, a persona,
    and a question, then must produce a faithful, persona-aligned answer.

    This is a single-step episode: the agent submits one answer and
    receives a terminal reward.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialise the answering environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._df: pd.DataFrame = pd.DataFrame()
        self._domain_name: str = ""
        self._domain_config = None
        self._persona = None
        self._question: str = ""

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self) -> AnsweringObservation:
        """Pick a random domain, persona, and question; load enriched data summary."""
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Pick random domain
        self._domain_name = random.choice(list(DOMAINS.keys()))
        self._domain_config = DOMAINS[self._domain_name]

        # Pick random persona
        self._persona = random.choice(PERSONAS)

        # Pick a domain-appropriate question
        self._question = random.choice(self._domain_config.example_questions)

        # Load data (simulate enriched data by loading raw)
        self._df = load_domain_data(self._domain_name, sample_size=100)

        # Compute dataset summary (basic stats for numeric columns)
        summary_parts = []
        for col in self._domain_config.numeric_columns:
            if col in self._df.columns:
                stats = self._df[col].describe()
                summary_parts.append(
                    f"{col}: mean={stats['mean']:.1f}, std={stats['std']:.1f}, "
                    f"min={stats['min']:.1f}, max={stats['max']:.1f}"
                )
        dataset_summary = "\n".join(summary_parts) if summary_parts else "No numeric summary available."

        # Compute column stats (first 12 columns)
        col_stats = []
        for col in self._df.columns[:12]:
            if self._df[col].dtype in ['int64', 'float64']:
                col_stats.append(f"{col}: {self._df[col].describe().to_dict()}")
            else:
                col_stats.append(f"{col}: {self._df[col].value_counts().head(5).to_dict()}")
        column_stats_str = "\n".join(col_stats)

        # Persona description
        persona_desc = (
            f"Role: {self._persona.role}. "
            f"Focus areas: {', '.join(self._persona.focus_areas)}. "
            f"Language style: {self._persona.language_style}."
        )

        return AnsweringObservation(
            domain=self._domain_name,
            dataset_summary=dataset_summary,
            persona=self._persona.name,
            persona_description=persona_desc,
            question=self._question,
            available_columns=list(self._df.columns),
            column_stats=column_stats_str,
            done=False,
            reward=0.0,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: AnsweringAction) -> AnsweringObservation:  # type: ignore[override]
        """Evaluate the answer and return a terminal observation with reward."""
        self._state.step_count += 1

        # Compute faithfulness
        faithfulness = self._compute_faithfulness(action, self._df)

        # Compute persona relevance
        persona_relevance = score_persona_alignment(action.answer, self._persona)

        # Optional: Patronus hallucination check
        context = (
            f"Domain: {self._domain_name}\n"
            f"Question: {self._question}\n"
            f"Available columns: {list(self._df.columns)}\n"
            f"Data sample:\n{format_preview(self._df)}"
        )
        patronus_score = self._get_patronus_score(action, context)

        # Compute final reward
        reward = answering_reward(faithfulness, persona_relevance, patronus_score)

        return AnsweringObservation(
            domain=self._domain_name,
            dataset_summary="",
            persona=self._persona.name,
            persona_description="",
            question=self._question,
            available_columns=list(self._df.columns),
            column_stats="",
            done=True,
            reward=reward,
            metadata={
                "faithfulness": faithfulness,
                "persona_relevance": persona_relevance,
                "patronus_score": patronus_score,
                "step": self._state.step_count,
            },
        )

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        """Return current environment state."""
        return self._state

    # ------------------------------------------------------------------
    # scoring helpers
    # ------------------------------------------------------------------
    def _compute_faithfulness(self, action: AnsweringAction, df: pd.DataFrame) -> float:
        """Score faithfulness based on cited columns and value references."""
        score = 0.0

        # Check cited columns exist
        valid_cols = [c for c in action.cited_columns if c in df.columns]
        if action.cited_columns:
            score += 0.5 * (len(valid_cols) / len(action.cited_columns))
        else:
            score += 0.1  # penalty for not citing

        # Check answer mentions real values
        answer_lower = action.answer.lower()
        for col in valid_cols[:3]:
            sample_vals = df[col].dropna().astype(str).head(10).tolist()
            if any(str(v).lower() in answer_lower for v in sample_vals):
                score += 0.15

        return min(score, 1.0)

    def _get_patronus_score(self, action: AnsweringAction, context: str):
        """Optionally call Patronus Lynx for hallucination checking."""
        api_key = os.environ.get("PATRONUS_API_KEY")
        if not api_key:
            return None
        try:
            import patronus
            patronus.init()
            from patronus import Patronus, RemoteEvaluator

            client = Patronus()
            lynx = RemoteEvaluator("lynx", "patronus:hallucination")
            result = client.evaluate(
                evaluators=lynx,
                task_output=action.answer,
                task_input=self._question,
                task_context=context,
            )
            return float(result.results[0].score)
        except Exception:
            return None
