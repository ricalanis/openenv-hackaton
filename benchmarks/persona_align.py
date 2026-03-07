"""PersonaAlign: Custom benchmark scorer for persona-tailored answers."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.shared.personas import PERSONAS, score_persona_alignment, Persona


class PersonaAlignScorer:
    """Scores answers against persona expectations."""

    def __init__(self):
        self.personas = {p.role: p for p in PERSONAS}

    def score(self, answer: str, persona_role: str) -> dict:
        """Score an answer for persona alignment.

        Returns dict with overall score and breakdown.
        """
        persona = self.personas.get(persona_role, PERSONAS[0])
        overall = score_persona_alignment(answer, persona)

        # Detailed breakdown
        answer_lower = answer.lower()
        keyword_hits = [kw for kw in persona.keywords if kw.lower() in answer_lower]
        anti_hits = [akw for akw in persona.anti_keywords if akw.lower() in answer_lower]

        return {
            "overall": overall,
            "keyword_hits": keyword_hits,
            "keyword_coverage": len(keyword_hits) / max(len(persona.keywords), 1),
            "anti_keyword_hits": anti_hits,
            "persona": persona_role,
        }

    def score_batch(self, answers: list[str], persona_role: str) -> dict:
        """Score a batch of answers."""
        scores = [self.score(a, persona_role) for a in answers]
        avg = sum(s["overall"] for s in scores) / max(len(scores), 1)
        return {
            "average_score": avg,
            "individual_scores": scores,
            "persona": persona_role,
            "n": len(answers),
        }


if __name__ == "__main__":
    scorer = PersonaAlignScorer()

    # Test examples
    exec_answer = "The Q3 revenue impact shows 15% year-over-year growth with cost reduction of $2M. ROI trends are positive."
    mgr_answer = "Team performance this sprint shows a bottleneck in the review process. Recommended action: increase capacity in QA."
    ic_answer = "My next step should be to finish the assigned task by Friday deadline. I need help understanding the priority."

    print("Executive:", scorer.score(exec_answer, "executive"))
    print("Manager:", scorer.score(mgr_answer, "manager"))
    print("IC:", scorer.score(ic_answer, "ic"))
