"""Reward computation with cached downstream signals."""

# Cached downstream signals mapping quality buckets to historical scores.
# These represent how well downstream stages perform given upstream quality.
DOWNSTREAM_CACHE: dict[str, float] = {
    "excellent": 0.95,  # DQ > 0.90 or coverage > 0.80
    "good": 0.75,       # DQ 0.70-0.90 or coverage 0.50-0.80
    "fair": 0.50,       # DQ 0.50-0.70 or coverage 0.30-0.50
    "poor": 0.20,       # DQ < 0.50 or coverage < 0.30
}


def _get_downstream_bucket(score: float) -> str:
    """Map a score to a downstream quality bucket."""
    if score > 0.90:
        return "excellent"
    elif score > 0.70:
        return "good"
    elif score > 0.50:
        return "fair"
    return "poor"


def cleaning_reward(dq_score: float, downstream_bucket: str = "") -> float:
    """Compute cleaning stage reward.

    0.70 * dq_score + 0.30 * downstream_signal
    """
    if not downstream_bucket:
        downstream_bucket = _get_downstream_bucket(dq_score)
    downstream = DOWNSTREAM_CACHE.get(downstream_bucket, 0.5)
    return round(0.70 * dq_score + 0.30 * downstream, 4)


def enrichment_reward(coverage: float, downstream_bucket: str = "") -> float:
    """Compute enrichment stage reward.

    0.50 * coverage + 0.50 * downstream_signal
    """
    if not downstream_bucket:
        downstream_bucket = _get_downstream_bucket(coverage)
    downstream = DOWNSTREAM_CACHE.get(downstream_bucket, 0.5)
    return round(0.50 * coverage + 0.50 * downstream, 4)


def answering_reward(faithfulness: float, persona_relevance: float,
                     patronus_score: float | None = None) -> float:
    """Compute answering stage reward.

    Without Patronus: 0.30 * faithfulness + 0.70 * persona_relevance
    With Patronus:    0.40 * patronus_faithfulness + 0.60 * persona_relevance
    """
    if patronus_score is not None:
        return round(0.40 * patronus_score + 0.60 * persona_relevance, 4)
    return round(0.30 * faithfulness + 0.70 * persona_relevance, 4)
