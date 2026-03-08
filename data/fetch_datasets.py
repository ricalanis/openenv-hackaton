"""Fetch real-world datasets, standardize, and upload to HuggingFace Hub.

Datasets (all open-licensed):
- HR: IBM HR Analytics Employee Attrition (Kaggle, ODbL, 1,470x35)
- Sales: CRM Sales Opportunities (Kaggle, Apache 2.0, ~8,800x12)
- IT Ops: UCI Incident Management Event Log (CC BY 4.0, sample 5,000 from 24,918)
- PM: Synthetic project management data (CC0-equivalent, 1,000+ rows)

License verification:
  HR     -> ODbL + DbCL (Open Database License) - commercial OK
  Sales  -> Apache 2.0 (Kaggle re-upload by innocentmfa) - commercial OK
  IT Ops -> CC BY 4.0 (UCI Machine Learning Repository) - commercial OK
  PM     -> Synthetic (no license restrictions) - commercial OK

Usage:
    python data/fetch_datasets.py
    python data/fetch_datasets.py --upload
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.labeling_functions import get_domain_lfs, majority_vote, GOOD, BAD


def fetch_hr() -> pd.DataFrame:
    """Fetch IBM HR Analytics dataset."""
    print("[HR] Fetching IBM HR Analytics dataset...")
    try:
        # Try kagglehub first
        import kagglehub
        path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
        import glob
        csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
        if csv_files:
            df = pd.read_csv(csv_files[0])
            print(f"[HR] Loaded from Kaggle: {df.shape}")
            return df
    except Exception as e:
        print(f"[HR] Kaggle download failed: {e}")

    # Fallback: try direct URL
    try:
        url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
        df = pd.read_csv(url)
        print(f"[HR] Loaded from GitHub: {df.shape}")
        return df
    except Exception as e:
        print(f"[HR] GitHub download failed: {e}")

    # Final fallback: generate synthetic
    print("[HR] Generating synthetic HR data...")
    return _generate_synthetic_hr()


def _generate_synthetic_hr(n: int = 1470) -> pd.DataFrame:
    """Generate synthetic HR data matching IBM schema."""
    rng = np.random.default_rng(42)
    departments = ["Sales", "Research & Development", "Human Resources"]
    job_roles = ["Sales Executive", "Research Scientist", "Manager",
                 "Lab Technician", "Manufacturing Director", "Healthcare Representative"]

    return pd.DataFrame({
        "EmployeeID": range(1, n + 1),
        "Age": rng.integers(18, 65, n),
        "Department": rng.choice(departments, n),
        "JobRole": rng.choice(job_roles, n),
        "MonthlyIncome": rng.integers(1000, 20000, n),
        "YearsAtCompany": rng.integers(0, 40, n),
        "Attrition": rng.choice(["Yes", "No"], n, p=[0.16, 0.84]),
        "JobSatisfaction": rng.integers(1, 5, n),
        "OverTime": rng.choice(["Yes", "No"], n, p=[0.28, 0.72]),
        "DistanceFromHome": rng.integers(1, 30, n),
        "Education": rng.integers(1, 5, n),
        "PerformanceRating": rng.choice([3, 4], n, p=[0.85, 0.15]),
        "EmployeeCount": np.ones(n, dtype=int),
        "StandardHours": np.full(n, 80, dtype=int),
        "Over18": ["Y"] * n,
    })


def fetch_sales() -> pd.DataFrame:
    """Fetch CRM Sales Opportunities dataset (Apache 2.0 license).

    Primary source: Kaggle innocentmfa/crm-sales-opportunities (Apache 2.0)
    Uses sales_pipeline.csv (8,800 rows) and maps columns to our schema.
    Fallback: synthetic data matching CRM schema
    """
    print("[Sales] Fetching CRM Sales dataset (Apache 2.0)...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("innocentmfa/crm-sales-opportunities")
        pipeline_file = os.path.join(path, "sales_pipeline.csv")
        # kagglehub may store in versions/ subdirectory
        if not os.path.exists(pipeline_file):
            import glob
            candidates = glob.glob(os.path.join(path, "**/sales_pipeline.csv"), recursive=True)
            if candidates:
                pipeline_file = candidates[0]
        if os.path.exists(pipeline_file):
            df = pd.read_csv(pipeline_file)
            df = _map_sales_columns(df)
            print(f"[Sales] Loaded from Kaggle (Apache 2.0): {df.shape}")
            return df
    except Exception as e:
        print(f"[Sales] Kaggle download failed: {e}")

    print("[Sales] Generating synthetic Sales data...")
    return _generate_synthetic_sales()


def _map_sales_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Kaggle CRM sales columns to our schema."""
    rng = np.random.default_rng(42)
    n = len(df)

    # Direct column mapping
    mapped = pd.DataFrame({
        "DealID": df.get("opportunity_id", pd.Series([f"D-{i:05d}" for i in range(n)])),
        "AccountName": df.get("account"),
        "Stage": df.get("deal_stage", pd.Series(["Prospecting"] * n)),
        "Amount": df.get("close_value", pd.Series(rng.lognormal(9, 1.5, n).astype(int))),
        "CloseDate": df.get("close_date"),
        "Rep": df.get("sales_agent"),
        "Product": df.get("product"),
    })

    # Synthesize missing columns that don't exist in the source
    regions = ["East", "West", "Central", "North", "South"]
    lead_sources = ["Website", "Referral", "Partner", "Trade Show", "Cold Call"]
    forecast_cats = ["Pipeline", "Best Case", "Commit", "Closed"]
    mapped["Region"] = rng.choice(regions, n)
    mapped["LeadSource"] = rng.choice(lead_sources, n)
    mapped["Probability"] = rng.uniform(0, 100, n).round(1)
    mapped["ForecastCategory"] = rng.choice(forecast_cats, n)

    # Compute DaysInStage from engage_date and close_date if available
    if "engage_date" in df.columns and "close_date" in df.columns:
        try:
            engage = pd.to_datetime(df["engage_date"], errors="coerce")
            close = pd.to_datetime(df["close_date"], errors="coerce")
            mapped["DaysInStage"] = (close - engage).dt.days.fillna(
                rng.integers(1, 120, n)).astype(int).clip(lower=1)
        except Exception:
            mapped["DaysInStage"] = rng.integers(1, 120, n)
    else:
        mapped["DaysInStage"] = rng.integers(1, 120, n)

    return mapped


def _generate_synthetic_sales(n: int = 8800) -> pd.DataFrame:
    """Generate synthetic sales data matching CRM schema."""
    rng = np.random.default_rng(42)
    stages = ["Prospecting", "Qualification", "Proposal", "Negotiation", "Won", "Lost"]
    products = ["GTX Pro", "GTX Basic", "GTX Plus", "MG Special", "MG Advanced"]
    regions = ["East", "West", "Central", "North", "South"]
    reps = [f"Rep_{i}" for i in range(1, 31)]
    lead_sources = ["Website", "Referral", "Partner", "Trade Show", "Cold Call"]
    forecast_cats = ["Pipeline", "Best Case", "Commit", "Closed"]

    stage_choices = rng.choice(stages, n, p=[0.15, 0.20, 0.20, 0.15, 0.20, 0.10])
    amounts = rng.lognormal(9, 1.5, n).astype(int)

    # Add nulls to mimic real data
    accounts = [f"Account_{i}" for i in rng.integers(1, 500, n)]
    # 1425 null accounts (matching real data pattern)
    null_account_idx = rng.choice(n, 1425, replace=False)
    for idx in null_account_idx:
        accounts[idx] = None

    close_dates = pd.date_range("2023-01-01", periods=n, freq="h").strftime("%Y-%m-%d").tolist()

    return pd.DataFrame({
        "DealID": [f"D-{i:05d}" for i in range(n)],
        "AccountName": accounts,
        "Stage": stage_choices,
        "Amount": amounts,
        "CloseDate": rng.choice(close_dates, n),
        "Rep": rng.choice(reps, n),
        "Product": rng.choice(products, n),
        "Region": rng.choice(regions, n),
        "LeadSource": rng.choice(lead_sources, n),
        "DaysInStage": rng.integers(1, 120, n),
        "Probability": rng.uniform(0, 100, n).round(1),
        "ForecastCategory": rng.choice(forecast_cats, n),
    })


def fetch_it_ops(sample_size: int = 5000) -> pd.DataFrame:
    """Fetch UCI Incident Management Event Log dataset (CC BY 4.0).

    Primary source: UCI ML Repository dataset #498
    License: Creative Commons Attribution 4.0 International
    Fallback: synthetic data matching incident management schema
    """
    print("[IT Ops] Fetching UCI Incident dataset (CC BY 4.0)...")
    try:
        import io
        import zipfile
        import urllib.request
        url = "https://archive.ics.uci.edu/static/public/498/incident+management+process+enriched+event+log.zip"
        print("[IT Ops] Downloading zip from UCI...")
        resp = urllib.request.urlopen(url, timeout=60)
        zip_data = io.BytesIO(resp.read())
        with zipfile.ZipFile(zip_data) as z:
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
            if csv_names:
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f, nrows=50000)
                # Group by incident_id and take first event per incident
                id_col = "number" if "number" in df.columns else df.columns[0]
                incidents = df.groupby(id_col).first().reset_index()
                if len(incidents) > sample_size:
                    incidents = incidents.sample(n=sample_size, random_state=42)
                incidents = _map_it_ops_columns(incidents)
                print(f"[IT Ops] Loaded from UCI (CC BY 4.0): {incidents.shape}")
                return incidents
    except Exception as e:
        print(f"[IT Ops] UCI download failed: {e}")

    print("[IT Ops] Generating synthetic IT Ops data...")
    return _generate_synthetic_it_ops(sample_size)


def _map_it_ops_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map UCI incident management columns to our schema."""
    rng = np.random.default_rng(42)
    n = len(df)

    # UCI columns: number, incident_state, active, reassignment_count,
    # reopen_count, sys_mod_count, made_sla, caller_id, opened_by,
    # opened_at, sys_created_by, sys_created_at, sys_updated_by,
    # sys_updated_at, contact_type, location, category, subcategory,
    # u_symptom, cmdb_ci, impact, urgency, priority, assignment_group,
    # assigned_to, knowledge, u_priority_confirmation, notify,
    # problem_id, rfc, vendor, caused_by, close_code, resolved_by,
    # resolved_at, closed_at

    mapped = pd.DataFrame()
    mapped["TicketID"] = df.get("number", pd.Series([f"INC-{i:06d}" for i in range(n)]))
    mapped["Category"] = df.get("category", pd.Series(["Software"] * n))
    mapped["Priority"] = df.get("priority", pd.Series(["3 - Moderate"] * n))
    mapped["Status"] = df.get("incident_state", pd.Series(["Open"] * n))
    mapped["Assignee"] = df.get("assigned_to", pd.Series([f"Agent_{i}" for i in range(n)]))
    mapped["CreatedDate"] = df.get("opened_at", df.get("sys_created_at"))
    mapped["ResolvedDate"] = df.get("resolved_at")
    mapped["AffectedSystem"] = df.get("cmdb_ci", pd.Series(["Unknown"] * n))
    mapped["ResolutionType"] = df.get("close_code", pd.Series(["Unknown"] * n))
    mapped["CustomerImpact"] = df.get("impact", pd.Series(["Medium"] * n))

    # Synthesize missing numeric columns
    sla_map = {"1 - Critical": 4, "2 - High": 8, "3 - Moderate": 24, "4 - Low": 48}
    if "priority" in df.columns:
        mapped["SLATarget"] = df["priority"].map(sla_map).fillna(24).astype(int)
    else:
        mapped["SLATarget"] = rng.choice([4, 8, 24, 48, 72], n)

    if "reassignment_count" in df.columns:
        mapped["EscalationLevel"] = pd.to_numeric(
            df["reassignment_count"], errors="coerce").fillna(0).astype(int).clip(upper=5)
    else:
        mapped["EscalationLevel"] = rng.integers(0, 5, n)

    return mapped


def _generate_synthetic_it_ops(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic IT operations incident data."""
    rng = np.random.default_rng(42)
    categories = ["Hardware", "Software", "Network", "Access", "Email"]
    priorities = ["P1-Critical", "P2-High", "P3-Medium", "P4-Low"]
    statuses = ["Open", "In Progress", "Resolved", "Closed", "Pending"]
    resolution_types = ["Fix Applied", "Workaround", "No Fix", "Duplicate", "User Error"]
    systems = ["ERP", "CRM", "Email Server", "VPN", "Database", "Web Portal", "Active Directory"]
    assignees = [f"Agent_{i}" for i in range(1, 51)]

    base_date = pd.Timestamp("2024-01-01")
    created = [base_date + pd.Timedelta(hours=int(h))
               for h in rng.integers(0, 8760, n)]

    status_choices = rng.choice(statuses, n, p=[0.10, 0.15, 0.35, 0.30, 0.10])
    resolved = []
    for i, s in enumerate(status_choices):
        if s in ("Resolved", "Closed"):
            delta = pd.Timedelta(hours=int(rng.integers(1, 168)))
            resolved.append((created[i] + delta).strftime("%Y-%m-%d %H:%M"))
        else:
            resolved.append(None)

    return pd.DataFrame({
        "TicketID": [f"INC-{i:06d}" for i in range(n)],
        "Category": rng.choice(categories, n),
        "Priority": rng.choice(priorities, n, p=[0.05, 0.15, 0.50, 0.30]),
        "Status": status_choices,
        "Assignee": rng.choice(assignees, n),
        "CreatedDate": [d.strftime("%Y-%m-%d %H:%M") for d in created],
        "ResolvedDate": resolved,
        "SLATarget": rng.choice([4, 8, 24, 48, 72], n, p=[0.05, 0.15, 0.40, 0.30, 0.10]),
        "EscalationLevel": rng.integers(0, 5, n),
        "AffectedSystem": rng.choice(systems, n),
        "ResolutionType": rng.choice(resolution_types, n),
        "CustomerImpact": rng.choice(["High", "Medium", "Low", "None"], n),
    })


def fetch_pm(augment_to: int = 1000) -> pd.DataFrame:
    """Fetch/generate Project Management dataset.

    Primary source: Kaggle programmer3/construction-project-management-dataset (CC0)
    Maps construction PM columns to our task-tracking schema.
    Fallback: synthetic PM data with augmentation to 1,000+ rows
    """
    print("[PM] Fetching Project Management dataset (CC0)...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("programmer3/construction-project-management-dataset")
        import glob
        csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
        if csv_files:
            df = pd.read_csv(csv_files[0])
            df = _map_pm_columns(df)
            print(f"[PM] Loaded from Kaggle (CC0): {df.shape}")
            return df
    except Exception as e:
        print(f"[PM] Kaggle download failed: {e}")

    print("[PM] Generating synthetic PM data...")
    df = _generate_synthetic_pm(300)

    # Augment to target size
    if len(df) < augment_to:
        augmented = _augment_pm(df, target_size=augment_to)
        print(f"[PM] Augmented: {df.shape} -> {augmented.shape}")
        return augmented
    return df


def _map_pm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Kaggle construction PM columns to our task-tracking schema."""
    rng = np.random.default_rng(42)
    n = len(df)

    # Source columns: Task_ID, Task_Duration_Days, Labor_Required, Equipment_Units,
    # Material_Cost_USD, Start_Constraint, Risk_Level, Resource_Constraint_Score,
    # Site_Constraint_Score, Dependency_Count

    projects = [f"Project_{chr(65 + i)}" for i in range(10)]
    statuses = ["Not Started", "In Progress", "Completed", "On Hold", "Cancelled"]
    priorities = ["Critical", "High", "Medium", "Low"]
    milestones = ["Planning", "Design", "Development", "Testing", "Deployment"]
    assignees = [f"Dev_{i}" for i in range(1, 21)]

    base_date = pd.Timestamp("2024-01-01")

    mapped = pd.DataFrame()
    mapped["TaskID"] = df.get("Task_ID", pd.Series([f"TASK-{i:04d}" for i in range(n)]))
    mapped["ProjectName"] = rng.choice(projects, n)
    mapped["Assignee"] = rng.choice(assignees, n)

    # Derive status from completion and risk
    mapped["Status"] = rng.choice(statuses, n, p=[0.10, 0.40, 0.30, 0.10, 0.10])
    mapped["Priority"] = rng.choice(priorities, n, p=[0.10, 0.25, 0.40, 0.25])

    # Map Risk_Level to RiskFlag
    if "Risk_Level" in df.columns:
        risk_map = {"High": "High", "Medium": "Medium", "Low": "Low"}
        mapped["RiskFlag"] = df["Risk_Level"].map(risk_map).fillna("None")
    else:
        mapped["RiskFlag"] = rng.choice(["High", "Medium", "Low", "None"], n)

    # Map duration to hours (8h per day)
    if "Task_Duration_Days" in df.columns:
        mapped["EstimatedHours"] = (df["Task_Duration_Days"] * 8).astype(int).clip(lower=2)
    else:
        mapped["EstimatedHours"] = rng.integers(2, 80, n)

    # ActualHours: derive from estimated with variance
    mapped["ActualHours"] = (mapped["EstimatedHours"] * rng.uniform(0.5, 1.5, n)).astype(int).clip(lower=0)

    # DueDate
    mapped["DueDate"] = [(base_date + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                         for d in rng.integers(0, 365, n)]

    # Dependencies from Dependency_Count
    if "Dependency_Count" in df.columns:
        deps = []
        for i, count in enumerate(df["Dependency_Count"]):
            try:
                c = int(count)
                if c > 0:
                    dep_ids = rng.integers(0, n, min(c, 3))
                    deps.append(",".join(f"TASK-{j:04d}" for j in dep_ids))
                else:
                    deps.append("")
            except (ValueError, TypeError):
                deps.append("")
        mapped["Dependencies"] = deps
    else:
        mapped["Dependencies"] = [""] * n

    mapped["Milestone"] = rng.choice(milestones, n)
    mapped["CompletionPct"] = rng.integers(0, 101, n)

    return mapped


def _generate_synthetic_pm(n: int = 300) -> pd.DataFrame:
    """Generate synthetic project management data."""
    rng = np.random.default_rng(42)
    projects = [f"Project_{chr(65 + i)}" for i in range(10)]
    statuses = ["Not Started", "In Progress", "Completed", "On Hold", "Cancelled"]
    priorities = ["Critical", "High", "Medium", "Low"]
    milestones = ["Planning", "Design", "Development", "Testing", "Deployment"]
    assignees = [f"Dev_{i}" for i in range(1, 21)]

    base_date = pd.Timestamp("2024-01-01")

    return pd.DataFrame({
        "TaskID": [f"TASK-{i:04d}" for i in range(n)],
        "ProjectName": rng.choice(projects, n),
        "Assignee": rng.choice(assignees, n),
        "Status": rng.choice(statuses, n, p=[0.10, 0.40, 0.30, 0.10, 0.10]),
        "Priority": rng.choice(priorities, n, p=[0.10, 0.25, 0.40, 0.25]),
        "DueDate": [(base_date + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                     for d in rng.integers(0, 365, n)],
        "EstimatedHours": rng.integers(2, 80, n),
        "ActualHours": rng.integers(0, 100, n),
        "Dependencies": [",".join(f"TASK-{j:04d}" for j in rng.integers(0, n, rng.integers(0, 3)))
                          if rng.random() > 0.3 else "" for _ in range(n)],
        "Milestone": rng.choice(milestones, n),
        "RiskFlag": rng.choice(["High", "Medium", "Low", "None"], n, p=[0.10, 0.20, 0.40, 0.30]),
        "CompletionPct": rng.integers(0, 101, n),
    })


def _augment_pm(df: pd.DataFrame, target_size: int = 1000) -> pd.DataFrame:
    """Augment PM data with synthetic variations."""
    rng = np.random.default_rng(42)
    rows_needed = target_size - len(df)
    if rows_needed <= 0:
        return df

    # Sample and perturb existing rows
    augmented_rows = []
    for _ in range(rows_needed):
        row = df.iloc[rng.integers(0, len(df))].copy()
        row_dict = row.to_dict()

        # Perturb numeric values
        for col in ["EstimatedHours", "ActualHours", "CompletionPct"]:
            if col in row_dict:
                try:
                    val = float(row_dict[col])
                    noise = rng.normal(0, val * 0.2) if val > 0 else rng.integers(1, 20)
                    row_dict[col] = max(0, int(val + noise))
                except (ValueError, TypeError):
                    pass

        if "CompletionPct" in row_dict:
            row_dict["CompletionPct"] = min(100, max(0, row_dict["CompletionPct"]))

        if "TaskID" in row_dict:
            row_dict["TaskID"] = f"TASK-{len(df) + len(augmented_rows):04d}"

        augmented_rows.append(row_dict)

    augmented_df = pd.DataFrame(augmented_rows)
    return pd.concat([df, augmented_df], ignore_index=True)


def standardize_columns(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Standardize column names and basic types for a domain."""
    from environments.shared.domains import DOMAINS
    config = DOMAINS[domain]

    # Map existing columns to expected names (case-insensitive matching)
    col_map = {}
    for expected in config.columns:
        for actual in df.columns:
            if actual.lower().replace("_", "") == expected.lower().replace("_", ""):
                col_map[actual] = expected
                break

    if col_map:
        df = df.rename(columns=col_map)

    # Ensure all expected columns exist
    for col in config.columns:
        if col not in df.columns:
            df[col] = None

    return df


def create_gold_standard(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Create gold-standard cleaned version of a domain dataset."""
    from environments.shared.domains import DOMAINS
    config = DOMAINS[domain]
    gold = df.copy()

    # Drop duplicates
    gold = gold.drop_duplicates()

    # Fix nulls in numeric columns with median
    for col in config.numeric_columns:
        if col in gold.columns:
            gold[col] = pd.to_numeric(gold[col], errors="coerce")
            median_val = gold[col].median()
            gold[col] = gold[col].fillna(median_val)

    # Standardize categorical columns
    for col in config.categorical_columns:
        if col in gold.columns:
            gold[col] = gold[col].astype(str).str.strip()

    return gold


def apply_labeling_functions(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Apply domain-specific labeling functions and add quality labels."""
    lfs = get_domain_lfs(domain)
    if not lfs:
        df["_dq_label"] = 0  # neutral
        return df

    labels = []
    for _, row in df.iterrows():
        label = majority_vote(row, lfs)
        labels.append(label)

    df = df.copy()
    df["_dq_label"] = labels
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch and prepare DataSage datasets")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace Hub")
    parser.add_argument("--output-dir", default="data/raw", help="Local output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = {}
    gold_datasets = {}

    # Fetch all domains
    fetchers = {
        "hr": fetch_hr,
        "sales": fetch_sales,
        "it_ops": fetch_it_ops,
        "pm": fetch_pm,
    }

    for domain, fetcher in fetchers.items():
        print(f"\n{'='*60}")
        print(f"Processing: {domain}")
        print(f"{'='*60}")

        df = fetcher()
        df = standardize_columns(df, domain)
        df = apply_labeling_functions(df, domain)

        # Save raw
        raw_path = os.path.join(args.output_dir, f"{domain}.parquet")
        df.to_parquet(raw_path, index=False)
        print(f"[{domain}] Saved raw: {raw_path} ({df.shape})")

        # Create and save gold standard
        gold = create_gold_standard(df, domain)
        gold_path = os.path.join(args.output_dir, f"{domain}_gold.parquet")
        gold.to_parquet(gold_path, index=False)
        print(f"[{domain}] Saved gold: {gold_path} ({gold.shape})")

        # Print DQ label distribution
        if "_dq_label" in df.columns:
            dist = df["_dq_label"].value_counts()
            print(f"[{domain}] DQ labels: {dict(dist)}")

        datasets[domain] = df
        gold_datasets[domain] = gold

    if args.upload:
        upload_to_hub(datasets, gold_datasets)

    print("\nDone! All datasets processed.")
    return datasets, gold_datasets


def upload_to_hub(datasets: dict, gold_datasets: dict):
    """Upload datasets to HuggingFace Hub.

    Each domain is uploaded as a separate config since schemas differ.
    Usage: load_dataset("ricalanis/datasage-enterprise-raw", "hr")
    """
    try:
        from datasets import Dataset

        print("\nUploading to HuggingFace Hub...")

        # Raw datasets — each domain as a separate config
        for domain, df in datasets.items():
            upload_df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
            ds = Dataset.from_pandas(upload_df)
            ds.push_to_hub("ricalanis/datasage-enterprise-raw", config_name=domain)
            print(f"  Uploaded raw/{domain}: {upload_df.shape}")
        print("Uploaded: ricalanis/datasage-enterprise-raw")

        # Gold datasets — each domain as a separate config
        for domain, df in gold_datasets.items():
            upload_df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
            ds = Dataset.from_pandas(upload_df)
            ds.push_to_hub("ricalanis/datasage-enterprise-gold", config_name=domain)
            print(f"  Uploaded gold/{domain}: {upload_df.shape}")
        print("Uploaded: ricalanis/datasage-enterprise-gold")

    except Exception as e:
        print(f"Upload failed: {e}")
        print("You can upload manually later with --upload flag after setting HF_TOKEN")


if __name__ == "__main__":
    main()
