"""Evaluate answering model on faithfulness and persona alignment."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.shared.domains import DOMAINS
from environments.shared.personas import PERSONAS, score_persona_alignment
from environments.shared.enterprise_data import load_domain_data


def eval_answering(model_name: str = None, domain: str = "hr",
                   persona: str = "executive", n_episodes: int = 5) -> dict:
    """Run answering evaluation for a domain x persona combination.

    Returns dict with: faithfulness, persona_alignment, patronus_score, combined_score.
    """
    config = DOMAINS[domain]
    persona_obj = next((p for p in PERSONAS if p.role == persona), PERSONAS[0])

    faithfulness_scores = []
    persona_scores = []
    patronus_scores = []

    for ep in range(n_episodes):
        df = load_domain_data(domain, sample_size=50)
        question = config.example_questions[ep % len(config.example_questions)]

        if model_name:
            # TODO: Load model and generate answer
            answer = f"Based on {config.columns[0]} analysis, the {domain} data shows trends."
            cited = config.columns[:3]
        else:
            # Baseline: template answer
            answer = f"The data shows various patterns in {domain}."
            cited = []

        # Faithfulness: check cited columns
        valid_cited = [c for c in cited if c in df.columns]
        faith = len(valid_cited) / max(len(cited), 1) if cited else 0.1
        faithfulness_scores.append(faith)

        # Persona alignment
        p_score = score_persona_alignment(answer, persona_obj)
        persona_scores.append(p_score)

        # Patronus (optional)
        patronus = _get_patronus_score(answer, question, df)
        if patronus is not None:
            patronus_scores.append(patronus)

    result = {
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "persona_alignment": sum(persona_scores) / len(persona_scores),
        "combined_score": 0.0,
        "episodes": n_episodes,
    }

    if patronus_scores:
        result["patronus_score"] = sum(patronus_scores) / len(patronus_scores)
        result["combined_score"] = 0.40 * result["patronus_score"] + 0.60 * result["persona_alignment"]
    else:
        result["patronus_score"] = None
        result["combined_score"] = 0.30 * result["faithfulness"] + 0.70 * result["persona_alignment"]

    return result


def _get_patronus_score(answer: str, question: str, df) -> float:
    """Get Patronus Lynx faithfulness score if API key available."""
    api_key = os.environ.get("PATRONUS_API_KEY")
    if not api_key:
        return None
    try:
        from patronus import Client
        client = Client(api_key=api_key)
        context = f"Question: {question}\nData columns: {list(df.columns)}"
        result = client.evaluate(
            evaluator="lynx-small",
            criteria="patronus:hallucination",
            evaluated_model_output=answer,
            task_context=context,
        )
        return float(result.score)
    except Exception:
        return None


if __name__ == "__main__":
    for domain in DOMAINS:
        for persona in ["executive", "manager", "ic"]:
            m = eval_answering(domain=domain, persona=persona)
            print(f"{domain}/{persona}: {m}")
