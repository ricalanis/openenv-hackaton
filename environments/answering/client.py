"""DataSage Answering Environment Client."""

from typing import Dict

import requests as _requests

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AnsweringAction, AnsweringObservation


class AnsweringEnv(
    EnvClient[AnsweringAction, AnsweringObservation, State]
):
    """
    Client for the DataSage Answering Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with AnsweringEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.question)
        ...
        ...     result = client.step(AnsweringAction(
        ...         answer="Based on the data...",
        ...         cited_columns=["col1"],
        ...         reasoning="Analysis reasoning"
        ...     ))
        ...     print(result.observation.reward)
    """

    def reset_with_seed(self, seed: int, domain: str | None = None) -> StepResult[AnsweringObservation]:
        """Reset with a specific seed for reproducible state."""
        payload = {"seed": seed}
        if domain:
            payload["domain"] = domain
        resp = _requests.post(f"{self.base_url}/reset-with-seed", json=payload)
        resp.raise_for_status()
        return self._parse_result(resp.json())

    def _step_payload(self, action: AnsweringAction) -> Dict:
        """
        Convert AnsweringAction to JSON payload for step message.

        Args:
            action: AnsweringAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "answer": action.answer,
            "cited_columns": action.cited_columns,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AnsweringObservation]:
        """
        Parse server response into StepResult[AnsweringObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with AnsweringObservation
        """
        obs_data = payload.get("observation", {})
        observation = AnsweringObservation(
            domain=obs_data.get("domain", ""),
            dataset_summary=obs_data.get("dataset_summary", ""),
            persona=obs_data.get("persona", ""),
            persona_description=obs_data.get("persona_description", ""),
            question=obs_data.get("question", ""),
            available_columns=obs_data.get("available_columns", []),
            column_stats=obs_data.get("column_stats", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
