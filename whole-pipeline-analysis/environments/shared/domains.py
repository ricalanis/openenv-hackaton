"""Domain registry for the 4 enterprise data domains."""

from pydantic import BaseModel


class DomainConfig(BaseModel):
    name: str
    display_name: str
    dataset_key: str
    columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    possible_enrichments: list[str]
    example_questions: list[str]


DOMAINS = {
    "hr": DomainConfig(
        name="hr",
        display_name="HR & People",
        dataset_key="hr",
        columns=[
            "EmployeeID", "Age", "Department", "JobRole", "MonthlyIncome",
            "YearsAtCompany", "Attrition", "JobSatisfaction", "OverTime",
            "DistanceFromHome", "Education", "PerformanceRating",
        ],
        numeric_columns=["Age", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome"],
        categorical_columns=["Department", "JobRole", "Attrition", "OverTime"],
        possible_enrichments=[
            "salary_band", "tenure_risk", "satisfaction_index",
            "industry_benchmark", "flight_risk_score",
        ],
        example_questions=[
            "Which departments have the highest attrition rates?",
            "What factors correlate most with employee turnover?",
            "How does overtime affect job satisfaction?",
            "What is the salary distribution across job roles?",
            "Which employees are at highest flight risk?",
        ],
    ),
    "sales": DomainConfig(
        name="sales",
        display_name="Sales & Revenue",
        dataset_key="sales",
        columns=[
            "DealID", "AccountName", "Stage", "Amount", "CloseDate",
            "Rep", "Product", "Region", "LeadSource", "DaysInStage",
            "Probability", "ForecastCategory",
        ],
        numeric_columns=["Amount", "DaysInStage", "Probability"],
        categorical_columns=["Stage", "Region", "Product", "ForecastCategory"],
        possible_enrichments=[
            "deal_size_category", "velocity_score", "win_probability_model",
            "industry_code", "competitive_risk",
        ],
        example_questions=[
            "What's our pipeline health for this quarter?",
            "Which deals are at risk of slipping?",
            "What's the average deal velocity by region?",
            "Which reps are below quota?",
            "What's the conversion rate by lead source?",
        ],
    ),
    "pm": DomainConfig(
        name="pm",
        display_name="Project Management",
        dataset_key="pm",
        columns=[
            "TaskID", "ProjectName", "Assignee", "Status", "Priority",
            "DueDate", "EstimatedHours", "ActualHours", "Dependencies",
            "Milestone", "RiskFlag", "CompletionPct",
        ],
        numeric_columns=["EstimatedHours", "ActualHours", "CompletionPct"],
        categorical_columns=["Status", "Priority", "RiskFlag"],
        possible_enrichments=[
            "schedule_risk_score", "resource_utilization",
            "dependency_chain_depth", "burndown_rate", "delay_probability",
        ],
        example_questions=[
            "Which projects are at risk of missing deadlines?",
            "How is resource utilization across teams?",
            "What's the burndown rate for the current sprint?",
            "Which tasks are blocking the most downstream work?",
            "What's our on-time delivery rate?",
        ],
    ),
    "it_ops": DomainConfig(
        name="it_ops",
        display_name="IT Operations",
        dataset_key="it_ops",
        columns=[
            "TicketID", "Category", "Priority", "Status", "Assignee",
            "CreatedDate", "ResolvedDate", "SLATarget", "EscalationLevel",
            "AffectedSystem", "ResolutionType", "CustomerImpact",
        ],
        numeric_columns=["SLATarget", "EscalationLevel"],
        categorical_columns=["Category", "Priority", "Status", "ResolutionType"],
        possible_enrichments=[
            "sla_compliance_flag", "mttr_band", "escalation_path",
            "incident_severity_score", "recurring_pattern_flag",
        ],
        example_questions=[
            "What's our SLA compliance rate this month?",
            "Which systems have the most incidents?",
            "What's the mean time to resolution trend?",
            "How many tickets are breaching SLA?",
            "What are the most common root causes?",
        ],
    ),
}
