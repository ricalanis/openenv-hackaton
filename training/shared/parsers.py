"""Parse model outputs into environment actions."""

import json
import re


def parse_cleaning_action(text: str) -> dict:
    """Parse model output into a CleaningAction dict."""
    # Try JSON extraction
    json_match = re.search(r'\{[^{}]*"operation"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "operation" in data and "column" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Fallback: keyword matching
    text_lower = text.lower()
    if "fill" in text_lower or "null" in text_lower:
        col = _extract_column(text)
        return {"operation": "fill_null", "column": col, "value": "median", "params": {}}
    elif "type" in text_lower or "cast" in text_lower or "convert" in text_lower:
        col = _extract_column(text)
        return {"operation": "fix_type", "column": col, "value": "numeric", "params": {}}
    elif "duplicate" in text_lower or "dedup" in text_lower:
        return {"operation": "remove_duplicate", "column": "", "params": {}}
    elif "standard" in text_lower or "normalize" in text_lower:
        col = _extract_column(text)
        return {"operation": "standardize", "column": col, "params": {}}
    elif "trim" in text_lower or "whitespace" in text_lower:
        col = _extract_column(text)
        return {"operation": "trim", "column": col, "params": {}}
    elif "typo" in text_lower or "correct" in text_lower:
        col = _extract_column(text)
        return {"operation": "correct_typo", "column": col, "params": {}}

    return {"operation": "fill_null", "column": "", "value": "median", "params": {}}


def parse_enrichment_action(text: str) -> dict:
    """Parse model output into an EnrichmentAction dict."""
    json_match = re.search(r'\{[^{}]*"operation"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "field_name" in data:
                return data
        except json.JSONDecodeError:
            pass

    text_lower = text.lower()
    # Try to find enrichment source name in text
    sources = ["salary_band", "tenure_risk", "satisfaction_index", "industry_benchmark",
               "flight_risk_score", "deal_size_category", "velocity_score",
               "win_probability_model", "industry_code", "competitive_risk",
               "schedule_risk_score", "resource_utilization", "dependency_chain_depth",
               "burndown_rate", "delay_probability", "sla_compliance_flag", "mttr_band",
               "escalation_path", "incident_severity_score", "recurring_pattern_flag"]

    for source in sources:
        if source.replace("_", " ") in text_lower or source in text_lower:
            return {"operation": "add_field", "field_name": source, "source": source,
                    "logic": "", "params": {}}

    return {"operation": "add_field", "field_name": "unknown", "source": "", "logic": "", "params": {}}


def parse_answering_action(text: str) -> dict:
    """Parse model output into an AnsweringAction dict."""
    json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "answer" in data:
                return data
        except json.JSONDecodeError:
            pass

    # For answering, the entire text IS the answer
    # Try to extract cited columns
    cited = re.findall(r'\b([A-Z][a-zA-Z]+(?:[A-Z][a-z]+)*)\b', text)

    return {
        "answer": text,
        "cited_columns": cited[:5],
        "reasoning": "",
    }


def _extract_column(text: str) -> str:
    """Try to extract a column name from text."""
    # Look for quoted column names
    quoted = re.findall(r'["\'](\w+)["\']', text)
    if quoted:
        return quoted[0]
    # Look for CamelCase words
    camel = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text)
    if camel:
        return camel[0]
    return ""
