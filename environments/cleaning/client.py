"""DataSage Cleaning Environment Client."""

from typing import Dict

import requests as _requests

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CleaningAction, CleaningObservation


class CleaningEnv(EnvClient[CleaningAction, CleaningObservation, State]):
    """
    Client for the DataSage Cleaning Environment.

    Example:
        >>> with CleaningEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.dq_score)
        ...     result = client.step(CleaningAction(
        ...         operation="fill_null", column="Age", value="median"
        ...     ))
        ...     print(result.observation.dq_score)
    """

    def reset_with_seed(self, seed: int, domain: str | None = None) -> StepResult[CleaningObservation]:
        """Reset with a specific seed for reproducible state."""
        payload = {"seed": seed}
        if domain:
            payload["domain"] = domain
        resp = _requests.post(f"{self.base_url}/reset-with-seed", json=payload)
        resp.raise_for_status()
        return self._parse_result(resp.json())

    def _step_payload(self, action: CleaningAction) -> Dict:
        """Convert CleaningAction to JSON payload."""
        return {
            "operation": action.operation,
            "column": action.column,
            "value": action.value,
            "params": action.params,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CleaningObservation]:
        """Parse server response into StepResult[CleaningObservation]."""
        obs_data = payload.get("observation", {})
        observation = CleaningObservation(
            domain=obs_data.get("domain", ""),
            data_preview=obs_data.get("data_preview", ""),
            dq_report=obs_data.get("dq_report", ""),
            dq_score=obs_data.get("dq_score", 0.0),
            columns_info=obs_data.get("columns_info", ""),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 15),
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
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
