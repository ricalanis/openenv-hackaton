"""DataSage Enrichment Environment Client."""

from typing import Dict

import requests as _requests

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EnrichmentAction, EnrichmentObservation


class EnrichmentEnv(
    EnvClient[EnrichmentAction, EnrichmentObservation, State]
):
    """
    Client for the DataSage Enrichment Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step enrichment interactions.

    Example:
        >>> with EnrichmentEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.domain)
        ...     print(result.observation.available_sources)
        ...
        ...     result = client.step(EnrichmentAction(
        ...         operation="add_field",
        ...         field_name="salary_band",
        ...         source="salary_band",
        ...     ))
        ...     print(result.observation.enrichment_coverage)
    """

    def reset_with_seed(self, seed: int, domain: str | None = None) -> StepResult[EnrichmentObservation]:
        """Reset with a specific seed for reproducible state."""
        payload = {"seed": seed}
        if domain:
            payload["domain"] = domain
        resp = _requests.post(f"{self.base_url}/reset-with-seed", json=payload)
        resp.raise_for_status()
        return self._parse_result(resp.json())

    def _step_payload(self, action: EnrichmentAction) -> Dict:
        """Convert EnrichmentAction to JSON payload for step message."""
        return {
            "operation": action.operation,
            "field_name": action.field_name,
            "source": action.source,
            "logic": action.logic,
            "params": action.params,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EnrichmentObservation]:
        """Parse server response into StepResult[EnrichmentObservation]."""
        obs_data = payload.get("observation", {})
        observation = EnrichmentObservation(
            domain=obs_data.get("domain", ""),
            data_preview=obs_data.get("data_preview", ""),
            schema_info=obs_data.get("schema_info", ""),
            available_sources=obs_data.get("available_sources", []),
            enrichment_coverage=obs_data.get("enrichment_coverage", 0.0),
            fields_added=obs_data.get("fields_added", []),
            possible_enrichments=obs_data.get("possible_enrichments", []),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 12),
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
