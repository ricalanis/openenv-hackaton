"""
DataSage Enrichment Environment Implementation.

An RL environment where an agent enriches enterprise datasets by adding
derived fields, lookups, and computed categories from a domain-specific
enrichment registry.
"""

import os
import random
import sys
from uuid import uuid4

# Allow imports from project root for shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from environments.shared.domains import DOMAINS
from environments.shared.enterprise_data import load_domain_data, format_preview
from environments.shared.enrichment_sources import (
    lookup,
    get_available_enrichments,
    get_enrichment_description,
    ENRICHMENT_REGISTRY,
)
from environments.shared.reward_utils import enrichment_reward

from models import EnrichmentAction, EnrichmentObservation
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State


class EnrichmentEnvironment(Environment):
    """
    Enrichment environment for the DataSage pipeline.

    The agent receives a cleaned dataset from a random enterprise domain and
    must enrich it by adding derived fields, lookups, and computed categories
    from the domain's enrichment registry. The goal is to maximise enrichment
    coverage (fraction of possible enrichments applied).

    Done condition: coverage > 0.80 or step_count >= 12.
    Reward: enrichment_reward(coverage, downstream_bucket).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the enrichment environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._domain: str = ""
        self._domain_config = None
        self._df = None
        self._fields_added: list[str] = []
        self._max_steps: int = 12

    def reset(self, seed: int | None = None, domain: str | None = None) -> EnrichmentObservation:
        """
        Reset the environment.

        Picks a random domain, loads a cleaned data batch (50 rows),
        and presents domain-specific enrichment sources.

        Args:
            seed: If provided, seeds ``random`` before any stochastic
                  operation so the reset is fully reproducible.
            domain: If provided (and valid), use this domain instead of a
                    random choice.

        Returns:
            EnrichmentObservation with initial dataset state and available enrichments.
        """
        # Seed RNG for reproducibility when requested
        if seed is not None:
            random.seed(seed)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._fields_added = []

        # Domain selection
        if domain is not None and domain in DOMAINS:
            self._domain = domain
        else:
            self._domain = random.choice(list(DOMAINS.keys()))
        self._domain_config = DOMAINS[self._domain]

        # Load cleaned data batch (50 rows)
        self._df = load_domain_data(self._domain, sample_size=50)

        # Build schema info
        schema_info = self._build_schema_info()

        # Available enrichment sources for this domain
        available = get_available_enrichments(self._domain)
        possible = list(self._domain_config.possible_enrichments)

        # Build enrichment descriptions for the observation
        source_descriptions = [
            f"{src}: {get_enrichment_description(self._domain, src)}"
            for src in available
        ]

        return EnrichmentObservation(
            domain=self._domain,
            data_preview=format_preview(self._df, n=5),
            schema_info=schema_info,
            available_sources=available,
            enrichment_coverage=0.0,
            fields_added=[],
            possible_enrichments=possible,
            step_number=0,
            max_steps=self._max_steps,
            done=False,
            reward=0.0,
            metadata={
                "domain_display": self._domain_config.display_name,
                "num_rows": len(self._df),
                "source_descriptions": source_descriptions,
            },
        )

    def step(self, action: EnrichmentAction) -> EnrichmentObservation:  # type: ignore[override]
        """
        Execute an enrichment step.

        Apply the requested enrichment by calling lookup(domain, source, row)
        for each row and adding the result as a new column.

        Args:
            action: EnrichmentAction describing the enrichment to apply.

        Returns:
            EnrichmentObservation with updated dataset state, coverage, and reward.
        """
        self._state.step_count += 1

        source = action.source or action.field_name
        field_name = action.field_name

        # Check if this enrichment source is valid for the domain
        available = get_available_enrichments(self._domain)
        error_msg = ""

        if source in available and field_name not in self._fields_added:
            # Apply enrichment: call lookup for each row and add as new column
            enriched_values = []
            for _, row in self._df.iterrows():
                val = lookup(self._domain, source, row.to_dict())
                enriched_values.append(val)

            self._df[field_name] = enriched_values
            self._fields_added.append(field_name)
        elif field_name in self._fields_added:
            error_msg = f"Field '{field_name}' already added."
        elif source not in available:
            error_msg = f"Source '{source}' not available. Choose from: {available}"

        # Compute coverage
        possible = self._domain_config.possible_enrichments
        coverage = len(self._fields_added) / len(possible) if possible else 0.0

        # Determine downstream bucket for reward
        if coverage > 0.80:
            downstream_bucket = "excellent"
        elif coverage > 0.50:
            downstream_bucket = "good"
        elif coverage > 0.30:
            downstream_bucket = "fair"
        else:
            downstream_bucket = "poor"

        # Compute reward
        reward = enrichment_reward(coverage, downstream_bucket)

        # Done condition
        done = coverage > 0.80 or self._state.step_count >= self._max_steps

        # Build schema info
        schema_info = self._build_schema_info()

        return EnrichmentObservation(
            domain=self._domain,
            data_preview=format_preview(self._df, n=5),
            schema_info=schema_info,
            available_sources=available,
            enrichment_coverage=round(coverage, 4),
            fields_added=list(self._fields_added),
            possible_enrichments=list(possible),
            step_number=self._state.step_count,
            max_steps=self._max_steps,
            done=done,
            reward=reward,
            metadata={
                "error": error_msg,
                "num_rows": len(self._df),
                "num_columns": len(self._df.columns),
            },
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def _build_schema_info(self) -> str:
        """Build a string describing the current DataFrame schema."""
        lines = []
        for col in self._df.columns:
            dtype = str(self._df[col].dtype)
            null_count = int(self._df[col].isnull().sum())
            lines.append(f"{col}: {dtype}, nulls={null_count}")
        return "\n".join(lines)
