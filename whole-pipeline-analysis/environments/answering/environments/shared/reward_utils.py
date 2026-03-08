"""Reward computation for DataSage training pipeline."""


def cleaning_reward(dq_before: float, dq_after: float) -> float:
    """Compute cleaning stage reward.

    Rewards both absolute DQ quality and the improvement delta.
    50% absolute quality + 50% improvement (scaled up to be meaningful).
    """
    improvement = max(0.0, dq_after - dq_before)
    return round(0.50 * dq_after + 0.50 * min(improvement * 5.0, 1.0), 4)


def enrichment_reward(coverage: float) -> float:
    """Compute enrichment stage reward.

    Direct coverage signal — no downstream mixing.
    """
    return round(coverage, 4)


def answering_reward(faithfulness: float, persona_relevance: float,
                     patronus_score: float | None = None) -> float:
    """Compute answering stage reward.

    Without Patronus: 0.30 * faithfulness + 0.70 * persona_relevance
    With Patronus:    0.40 * patronus_score + 0.60 * persona_relevance
    """
    if patronus_score is not None:
        return round(0.40 * patronus_score + 0.60 * persona_relevance, 4)
    return round(0.30 * faithfulness + 0.70 * persona_relevance, 4)
