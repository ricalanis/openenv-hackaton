"""Static enrichment lookup tables per domain (no API calls)."""

import numpy as np

# Enrichment registry: domain -> source -> lookup function or static data
ENRICHMENT_REGISTRY: dict[str, dict[str, dict]] = {
    "hr": {
        "salary_band": {
            "description": "BLS salary band classification based on monthly income",
            "type": "derived",
            "logic": "classify_salary_band",
        },
        "tenure_risk": {
            "description": "Tenure-based flight risk score",
            "type": "derived",
            "logic": "compute_tenure_risk",
        },
        "satisfaction_index": {
            "description": "Composite satisfaction index from multiple factors",
            "type": "derived",
            "logic": "compute_satisfaction_index",
        },
        "industry_benchmark": {
            "description": "Industry benchmark salary percentile",
            "type": "lookup",
            "data": {
                "Sales Executive": 65000, "Research Scientist": 72000,
                "Manager": 85000, "Lab Technician": 45000,
                "Manufacturing Director": 95000, "Healthcare Representative": 55000,
                "Human Resources": 60000,
            },
        },
        "flight_risk_score": {
            "description": "Combined flight risk from satisfaction, tenure, overtime",
            "type": "derived",
            "logic": "compute_flight_risk",
        },
    },
    "sales": {
        "deal_size_category": {
            "description": "Categorize deal by amount: Small/Medium/Large/Enterprise",
            "type": "derived",
            "logic": "classify_deal_size",
        },
        "velocity_score": {
            "description": "Deal velocity based on days in stage vs benchmark",
            "type": "derived",
            "logic": "compute_velocity_score",
        },
        "win_probability_model": {
            "description": "Heuristic win probability based on stage + days",
            "type": "derived",
            "logic": "compute_win_probability",
        },
        "industry_code": {
            "description": "Industry classification code from account name patterns",
            "type": "lookup",
            "data": {
                "Tech": "SIC-7372", "Healthcare": "SIC-8011",
                "Finance": "SIC-6020", "Retail": "SIC-5311",
                "Manufacturing": "SIC-3559", "default": "SIC-9999",
            },
        },
        "competitive_risk": {
            "description": "Competitive risk score based on deal stage and velocity",
            "type": "derived",
            "logic": "compute_competitive_risk",
        },
    },
    "pm": {
        "schedule_risk_score": {
            "description": "Risk of schedule slippage based on progress vs due date",
            "type": "derived",
            "logic": "compute_schedule_risk",
        },
        "resource_utilization": {
            "description": "Resource utilization ratio: actual/estimated hours",
            "type": "derived",
            "logic": "compute_resource_utilization",
        },
        "dependency_chain_depth": {
            "description": "Depth of dependency chain for task",
            "type": "derived",
            "logic": "compute_dependency_depth",
        },
        "burndown_rate": {
            "description": "Task completion rate relative to plan",
            "type": "derived",
            "logic": "compute_burndown_rate",
        },
        "delay_probability": {
            "description": "Probability of delay based on current trajectory",
            "type": "derived",
            "logic": "compute_delay_probability",
        },
    },
    "it_ops": {
        "sla_compliance_flag": {
            "description": "Whether ticket meets SLA target",
            "type": "derived",
            "logic": "compute_sla_compliance",
        },
        "mttr_band": {
            "description": "Mean time to resolution band: Fast/Normal/Slow/Critical",
            "type": "derived",
            "logic": "classify_mttr",
        },
        "escalation_path": {
            "description": "Recommended escalation path based on category and priority",
            "type": "lookup",
            "data": {
                "P1-Critical": "L3 -> Manager -> VP",
                "P2-High": "L2 -> L3 -> Manager",
                "P3-Medium": "L1 -> L2",
                "P4-Low": "L1",
            },
        },
        "incident_severity_score": {
            "description": "Computed severity score from priority and customer impact",
            "type": "derived",
            "logic": "compute_severity_score",
        },
        "recurring_pattern_flag": {
            "description": "Flag indicating likely recurring issue",
            "type": "derived",
            "logic": "detect_recurring_pattern",
        },
    },
}


def lookup(domain: str, source: str, row: dict) -> object:
    """Unified lookup/compute function for enrichment values."""
    registry = ENRICHMENT_REGISTRY.get(domain, {})
    source_config = registry.get(source)
    if not source_config:
        return None

    if source_config["type"] == "lookup":
        # Direct lookup from static data
        data = source_config["data"]
        # Try various keys from the row
        for key_col in row:
            val = str(row.get(key_col, ""))
            if val in data:
                return data[val]
        return data.get("default")

    # Derived computations
    logic = source_config["logic"]
    compute_fn = _COMPUTE_FUNCTIONS.get(logic)
    if compute_fn:
        return compute_fn(row)
    return None


# --- Computation functions ---

def _classify_salary_band(row: dict) -> str:
    try:
        income = float(row.get("MonthlyIncome", 0))
    except (ValueError, TypeError):
        return "Unknown"
    if income < 3000:
        return "Entry"
    elif income < 6000:
        return "Mid"
    elif income < 10000:
        return "Senior"
    return "Executive"


def _compute_tenure_risk(row: dict) -> float:
    try:
        years = float(row.get("YearsAtCompany", 0))
    except (ValueError, TypeError):
        return 0.5
    # Short tenure = higher risk, very long = moderate risk
    if years < 2:
        return 0.8
    elif years < 5:
        return 0.4
    elif years < 10:
        return 0.2
    return 0.3


def _compute_satisfaction_index(row: dict) -> float:
    try:
        satisfaction = float(row.get("JobSatisfaction", 3))
    except (ValueError, TypeError):
        satisfaction = 3
    return round(satisfaction / 4.0, 2)


def _compute_flight_risk(row: dict) -> float:
    tenure_risk = _compute_tenure_risk(row)
    sat_index = _compute_satisfaction_index(row)
    overtime = 0.3 if str(row.get("OverTime", "No")).lower() == "yes" else 0.0
    return round(0.4 * tenure_risk + 0.4 * (1 - sat_index) + 0.2 * overtime, 2)


def _classify_deal_size(row: dict) -> str:
    try:
        amount = float(row.get("Amount", 0))
    except (ValueError, TypeError):
        return "Unknown"
    if amount < 5000:
        return "Small"
    elif amount < 25000:
        return "Medium"
    elif amount < 100000:
        return "Large"
    return "Enterprise"


def _compute_velocity_score(row: dict) -> float:
    try:
        days = float(row.get("DaysInStage", 0))
    except (ValueError, TypeError):
        return 0.5
    # Benchmark: 30 days per stage
    if days < 15:
        return 1.0
    elif days < 30:
        return 0.7
    elif days < 60:
        return 0.4
    return 0.1


def _compute_win_probability(row: dict) -> float:
    stage_probs = {
        "Prospecting": 0.10, "Qualification": 0.25, "Proposal": 0.50,
        "Negotiation": 0.75, "Won": 1.0, "Lost": 0.0,
    }
    stage = str(row.get("Stage", ""))
    base_prob = stage_probs.get(stage, 0.3)
    velocity = _compute_velocity_score(row)
    return round(0.7 * base_prob + 0.3 * velocity, 2)


def _compute_competitive_risk(row: dict) -> float:
    velocity = _compute_velocity_score(row)
    return round(1.0 - velocity, 2)


def _compute_schedule_risk(row: dict) -> float:
    try:
        pct = float(row.get("CompletionPct", 0))
    except (ValueError, TypeError):
        pct = 0
    # Simple: lower completion = higher risk
    return round(1.0 - (pct / 100.0), 2)


def _compute_resource_utilization(row: dict) -> float:
    try:
        estimated = float(row.get("EstimatedHours", 1))
        actual = float(row.get("ActualHours", 0))
    except (ValueError, TypeError):
        return 0.0
    if estimated == 0:
        return 0.0
    return round(actual / estimated, 2)


def _compute_dependency_depth(row: dict) -> int:
    deps = row.get("Dependencies", "")
    if not deps or str(deps) in ("nan", "None", ""):
        return 0
    return len(str(deps).split(","))


def _compute_burndown_rate(row: dict) -> float:
    try:
        pct = float(row.get("CompletionPct", 0))
        estimated = float(row.get("EstimatedHours", 1))
        actual = float(row.get("ActualHours", 0))
    except (ValueError, TypeError):
        return 0.5
    if actual == 0:
        return 0.0
    expected_rate = pct / 100.0
    time_rate = actual / max(estimated, 1)
    return round(expected_rate / max(time_rate, 0.01), 2)


def _compute_delay_probability(row: dict) -> float:
    schedule_risk = _compute_schedule_risk(row)
    burndown = _compute_burndown_rate(row)
    return round(schedule_risk * (1.0 / max(burndown, 0.1)), 2)


def _compute_sla_compliance(row: dict) -> str:
    try:
        sla = float(row.get("SLATarget", 24))
        escalation = float(row.get("EscalationLevel", 0))
    except (ValueError, TypeError):
        return "Unknown"
    if escalation > 2:
        return "Breached"
    return "Compliant"


def _classify_mttr(row: dict) -> str:
    try:
        escalation = float(row.get("EscalationLevel", 0))
    except (ValueError, TypeError):
        return "Normal"
    if escalation == 0:
        return "Fast"
    elif escalation <= 1:
        return "Normal"
    elif escalation <= 3:
        return "Slow"
    return "Critical"


def _compute_severity_score(row: dict) -> float:
    priority_scores = {"P1-Critical": 1.0, "P2-High": 0.7, "P3-Medium": 0.4, "P4-Low": 0.1}
    priority = str(row.get("Priority", "P3-Medium"))
    return priority_scores.get(priority, 0.4)


def _detect_recurring_pattern(row: dict) -> bool:
    category = str(row.get("Category", ""))
    # Simple heuristic: certain categories tend to recur
    recurring_cats = {"Network", "Email", "Access"}
    return category in recurring_cats


_COMPUTE_FUNCTIONS = {
    "classify_salary_band": _classify_salary_band,
    "compute_tenure_risk": _compute_tenure_risk,
    "compute_satisfaction_index": _compute_satisfaction_index,
    "compute_flight_risk": _compute_flight_risk,
    "classify_deal_size": _classify_deal_size,
    "compute_velocity_score": _compute_velocity_score,
    "compute_win_probability": _compute_win_probability,
    "compute_competitive_risk": _compute_competitive_risk,
    "compute_schedule_risk": _compute_schedule_risk,
    "compute_resource_utilization": _compute_resource_utilization,
    "compute_dependency_depth": _compute_dependency_depth,
    "compute_burndown_rate": _compute_burndown_rate,
    "compute_delay_probability": _compute_delay_probability,
    "compute_sla_compliance": _compute_sla_compliance,
    "classify_mttr": _classify_mttr,
    "compute_severity_score": _compute_severity_score,
    "detect_recurring_pattern": _detect_recurring_pattern,
}


def get_available_enrichments(domain: str) -> list[str]:
    """Return list of available enrichment source names for a domain."""
    return list(ENRICHMENT_REGISTRY.get(domain, {}).keys())


def get_enrichment_description(domain: str, source: str) -> str:
    """Get human-readable description of an enrichment source."""
    registry = ENRICHMENT_REGISTRY.get(domain, {})
    config = registry.get(source, {})
    return config.get("description", "Unknown enrichment source")
