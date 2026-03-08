"""Generalized personas for domain-independent question answering."""

import re

from pydantic import BaseModel


class Persona(BaseModel):
    name: str
    role: str
    focus_areas: list[str]
    language_style: str
    keywords: list[str]
    anti_keywords: list[str]


PERSONAS = [
    Persona(
        name="Executive",
        role="executive",
        focus_areas=["costs", "ROI", "strategic risk", "portfolio trends", "year-over-year"],
        language_style="strategic-financial",
        keywords=["revenue", "cost", "ROI", "risk", "trend", "quarter",
                   "year-over-year", "impact", "budget", "margin", "growth"],
        anti_keywords=["I think", "maybe", "um", "idk"],
    ),
    Persona(
        name="Manager",
        role="manager",
        focus_areas=["team performance", "operational health", "process bottlenecks", "capacity"],
        language_style="operational-actionable",
        keywords=["team", "performance", "bottleneck", "capacity", "SLA",
                   "process", "action", "priority", "escalation", "delivery"],
        anti_keywords=["shareholder", "valuation", "IPO"],
    ),
    Persona(
        name="Individual Contributor",
        role="ic",
        focus_areas=["personal tasks", "deadlines", "what to do next", "simple explanations"],
        language_style="plain-personal",
        keywords=["my", "I should", "next step", "deadline", "help",
                   "understand", "priority", "task", "assigned"],
        anti_keywords=["KPI", "ROI", "portfolio", "strategic", "EBITDA"],
    ),
]

PERSONA_MAP = {p.role: p for p in PERSONAS}


def get_persona(role: str) -> Persona:
    """Get a persona by role name."""
    return PERSONA_MAP[role]


def score_persona_alignment(answer: str, persona: Persona) -> float:
    """Score how well an answer aligns with a persona's communication style.

    Returns a float 0-1 based on:
    - Keyword density (presence of expected terms)
    - Anti-keyword penalty (presence of terms that don't fit)
    - Formality check (matches language style)
    """
    answer_lower = answer.lower()
    words = re.findall(r'\w+', answer_lower)
    word_count = max(len(words), 1)

    # Keyword scoring: fraction of persona keywords found
    keyword_hits = sum(1 for kw in persona.keywords if kw.lower() in answer_lower)
    keyword_score = min(keyword_hits / max(len(persona.keywords) * 0.3, 1), 1.0)

    # Anti-keyword penalty
    anti_hits = sum(1 for akw in persona.anti_keywords if akw.lower() in answer_lower)
    anti_penalty = min(anti_hits * 0.15, 0.5)

    # Formality check
    formality_score = _check_formality(answer, persona.language_style)

    # Combine: 50% keywords, 20% formality, 30% base (minus anti-penalty)
    raw_score = 0.50 * keyword_score + 0.20 * formality_score + 0.30 - anti_penalty
    return round(max(0.0, min(1.0, raw_score)), 4)


def _check_formality(text: str, style: str) -> float:
    """Check if text formality matches the expected language style."""
    text_lower = text.lower()

    if style == "strategic-financial":
        indicators = ["percent", "%", "million", "billion", "quarter", "fiscal",
                       "forecast", "benchmark", "metric"]
        hits = sum(1 for ind in indicators if ind in text_lower)
        return min(hits / 2.0, 1.0)

    elif style == "operational-actionable":
        indicators = ["action", "recommend", "should", "priority", "next steps",
                       "immediate", "plan", "schedule"]
        hits = sum(1 for ind in indicators if ind in text_lower)
        return min(hits / 2.0, 1.0)

    elif style == "plain-personal":
        # Plain style rewards shorter sentences and simple language
        sentences = text.split(".")
        avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        return 1.0 if avg_len < 20 else max(0.0, 1.0 - (avg_len - 20) / 30)

    return 0.5
