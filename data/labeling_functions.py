"""Snorkel-style labeling functions for programmatic data quality assessment.

Each labeling function returns:
    ABSTAIN = -1 (no opinion)
    BAD = 0     (row has quality issue)
    GOOD = 1    (row passes this check)
"""

import pandas as pd

ABSTAIN, BAD, GOOD = -1, 0, 1


# ── HR Domain LFs ────────────────────────────────────────────────────

def lf_hr_constant_columns(row):
    """Flag rows where known-constant columns have unexpected values."""
    # EmployeeCount should always be 1, StandardHours=80, Over18=Y
    ec = row.get("EmployeeCount")
    if ec is not None and not pd.isna(ec):
        try:
            if int(ec) != 1:
                return BAD
        except (ValueError, TypeError):
            return BAD
    sh = row.get("StandardHours")
    if sh is not None and not pd.isna(sh):
        try:
            if int(sh) != 80:
                return BAD
        except (ValueError, TypeError):
            return BAD
    return ABSTAIN


def lf_hr_salary_range(row):
    """MonthlyIncome should be between 1000 and 20000."""
    income = row.get("MonthlyIncome")
    if income is None or pd.isna(income):
        return BAD
    try:
        val = float(income)
        return GOOD if 1000 <= val <= 20000 else BAD
    except (ValueError, TypeError):
        return BAD


def lf_hr_age_range(row):
    """Age should be between 18 and 65."""
    age = row.get("Age")
    if age is None or pd.isna(age):
        return BAD
    try:
        val = float(age)
        return GOOD if 18 <= val <= 65 else BAD
    except (ValueError, TypeError):
        return BAD


def lf_hr_satisfaction_range(row):
    """JobSatisfaction should be 1-4."""
    sat = row.get("JobSatisfaction")
    if sat is None or pd.isna(sat):
        return ABSTAIN
    try:
        val = int(float(sat))
        return GOOD if 1 <= val <= 4 else BAD
    except (ValueError, TypeError):
        return BAD


def lf_hr_years_nonnegative(row):
    """YearsAtCompany should be non-negative."""
    years = row.get("YearsAtCompany")
    if years is None or pd.isna(years):
        return BAD
    try:
        return GOOD if float(years) >= 0 else BAD
    except (ValueError, TypeError):
        return BAD


# ── Sales Domain LFs ─────────────────────────────────────────────────

def lf_sales_null_account(row):
    """Engaging/Prospecting stages may lack account but Won stages must have one."""
    stage = str(row.get("Stage", ""))
    account = row.get("AccountName")
    if stage in ("Won", "Negotiation") and (account is None or pd.isna(account) or str(account).strip() == ""):
        return BAD
    if account is not None and not pd.isna(account) and str(account).strip() != "":
        return GOOD
    return ABSTAIN


def lf_sales_spelling(row):
    """Detect common spelling inconsistencies like GTXPro vs GTX Pro."""
    product = str(row.get("Product", ""))
    known_bad = ["GTXPro", "GTXBasic", "GTXPlus", "MGSpecial", "MGAdvanced"]
    if product in known_bad:
        return BAD
    return ABSTAIN


def lf_sales_close_value_stage(row):
    """Won deals must have a positive Amount."""
    stage = str(row.get("Stage", ""))
    amount = row.get("Amount")
    if stage == "Won":
        if amount is None or pd.isna(amount):
            return BAD
        try:
            return GOOD if float(amount) > 0 else BAD
        except (ValueError, TypeError):
            return BAD
    return ABSTAIN


def lf_sales_probability_range(row):
    """Probability should be between 0 and 1 (or 0-100)."""
    prob = row.get("Probability")
    if prob is None or pd.isna(prob):
        return ABSTAIN
    try:
        val = float(prob)
        return GOOD if 0 <= val <= 100 else BAD
    except (ValueError, TypeError):
        return BAD


def lf_sales_amount_positive(row):
    """Deal Amount should be non-negative."""
    amount = row.get("Amount")
    if amount is None or pd.isna(amount):
        return ABSTAIN
    try:
        return GOOD if float(amount) >= 0 else BAD
    except (ValueError, TypeError):
        return BAD


# ── IT Ops Domain LFs ────────────────────────────────────────────────

def lf_itops_sla_breach(row):
    """Flag tickets where resolution likely breaches SLA."""
    resolved = row.get("ResolvedDate")
    created = row.get("CreatedDate")
    sla = row.get("SLATarget")
    if resolved is None or pd.isna(resolved):
        return ABSTAIN
    if sla is not None:
        try:
            if float(sla) < 0:
                return BAD
        except (ValueError, TypeError):
            return BAD
    return GOOD


def lf_itops_missing_resolution(row):
    """Closed/Resolved tickets must have a ResolvedDate."""
    status = str(row.get("Status", ""))
    resolved = row.get("ResolvedDate")
    if status in ("Closed", "Resolved"):
        if resolved is None or pd.isna(resolved) or str(resolved).strip() == "":
            return BAD
        return GOOD
    return ABSTAIN


def lf_itops_reassignment_count(row):
    """High escalation level suggests complexity."""
    escalation = row.get("EscalationLevel")
    if escalation is None or pd.isna(escalation):
        return ABSTAIN
    try:
        val = int(float(escalation))
        return BAD if val > 5 else GOOD
    except (ValueError, TypeError):
        return BAD


def lf_itops_priority_valid(row):
    """Priority should be one of the expected values."""
    priority = str(row.get("Priority", ""))
    valid = {"P1-Critical", "P2-High", "P3-Medium", "P4-Low"}
    if priority in valid:
        return GOOD
    if priority.strip() == "":
        return ABSTAIN
    return BAD


# ── PM Domain LFs ────────────────────────────────────────────────────

def lf_pm_budget_overrun(row):
    """Flag tasks where actual hours greatly exceed estimated."""
    try:
        actual = float(row.get("ActualHours", 0))
        estimated = float(row.get("EstimatedHours", 0))
    except (ValueError, TypeError):
        return BAD
    if estimated > 0 and actual > estimated * 2:
        return BAD
    if estimated > 0 and actual <= estimated:
        return GOOD
    return ABSTAIN


def lf_pm_progress_status_mismatch(row):
    """'Not Started' tasks should have 0% completion."""
    status = str(row.get("Status", ""))
    try:
        pct = float(row.get("CompletionPct", 0))
    except (ValueError, TypeError):
        return BAD
    if status == "Not Started" and pct > 0:
        return BAD
    if status == "Completed" and pct < 100:
        return BAD
    return GOOD


def lf_pm_completion_range(row):
    """CompletionPct should be between 0 and 100."""
    pct = row.get("CompletionPct")
    if pct is None or pd.isna(pct):
        return BAD
    try:
        val = float(pct)
        return GOOD if 0 <= val <= 100 else BAD
    except (ValueError, TypeError):
        return BAD


def lf_pm_hours_positive(row):
    """Hours should be non-negative."""
    for col in ("EstimatedHours", "ActualHours"):
        val = row.get(col)
        if val is not None and not pd.isna(val):
            try:
                if float(val) < 0:
                    return BAD
            except (ValueError, TypeError):
                return BAD
    return GOOD


# ── Generic LFs (apply to all domains) ──────────────────────────────

def lf_has_critical_nulls(row, critical_cols=None):
    """Flag rows where first 3 domain columns are null."""
    if critical_cols is None:
        return ABSTAIN
    for col in critical_cols:
        if col in row and pd.isna(row[col]):
            return BAD
    return ABSTAIN


def lf_type_mismatch_generic(row, numeric_cols=None):
    """Flag rows where numeric fields contain non-numeric values."""
    if numeric_cols is None:
        return ABSTAIN
    for col in numeric_cols:
        val = row.get(col)
        if val is not None and not pd.isna(val):
            try:
                float(val)
            except (ValueError, TypeError):
                return BAD
    return ABSTAIN


def lf_all_fields_valid(row, required_cols=None):
    """Positive signal: all required fields present and well-typed."""
    if required_cols is None:
        return ABSTAIN
    for col in required_cols:
        if col in row and pd.isna(row[col]):
            return ABSTAIN
    return GOOD


# ── Domain LF registry ──────────────────────────────────────────────

_DOMAIN_LFS = {
    "hr": [lf_hr_constant_columns, lf_hr_salary_range, lf_hr_age_range,
           lf_hr_satisfaction_range, lf_hr_years_nonnegative],
    "sales": [lf_sales_null_account, lf_sales_spelling, lf_sales_close_value_stage,
              lf_sales_probability_range, lf_sales_amount_positive],
    "it_ops": [lf_itops_sla_breach, lf_itops_missing_resolution,
               lf_itops_reassignment_count, lf_itops_priority_valid],
    "pm": [lf_pm_budget_overrun, lf_pm_progress_status_mismatch,
           lf_pm_completion_range, lf_pm_hours_positive],
}


def get_domain_lfs(domain: str) -> list:
    """Get labeling functions for a specific domain."""
    return _DOMAIN_LFS.get(domain, [])


def majority_vote(row, lfs: list) -> int:
    """Simple majority vote aggregation across labeling functions.

    Returns GOOD if majority vote GOOD, BAD if majority BAD, ABSTAIN otherwise.
    """
    votes = []
    for lf in lfs:
        try:
            vote = lf(row)
            if vote != ABSTAIN:
                votes.append(vote)
        except Exception:
            continue

    if not votes:
        return ABSTAIN

    good_count = sum(1 for v in votes if v == GOOD)
    bad_count = sum(1 for v in votes if v == BAD)

    if good_count > bad_count:
        return GOOD
    elif bad_count > good_count:
        return BAD
    return ABSTAIN
