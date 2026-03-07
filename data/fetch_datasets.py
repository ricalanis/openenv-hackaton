"""Fetch real-world datasets, standardize, and upload to HuggingFace Hub.

Datasets:
- HR: IBM HR Analytics Employee Attrition (Kaggle, 1,470×35)
- Sales: Maven Analytics CRM Sales Opportunities (8,800×18)
- IT Ops: UCI Incident Management Event Log (sample 5,000 from 24,918)
- PM: JohnVans123/ProjectManagement (HF) + synthetic augmentation to 1,000+

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
    """Fetch Maven Analytics CRM Sales Opportunities dataset."""
    print("[Sales] Fetching CRM Sales dataset...")
    try:
        # Try loading from a common public source
        urls = [
            "https://raw.githubusercontent.com/datasets/crm-sales-opportunities/main/data.csv",
        ]
        for url in urls:
            try:
                df = pd.read_csv(url)
                print(f"[Sales] Loaded from URL: {df.shape}")
                return df
            except Exception:
                continue
    except Exception as e:
        print(f"[Sales] Download failed: {e}")

    print("[Sales] Generating synthetic Sales data...")
    return _generate_synthetic_sales()


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
    """Fetch UCI Incident Management Event Log dataset."""
    print("[IT Ops] Fetching UCI Incident dataset...")
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00498/incident_event_log.csv"
        df = pd.read_csv(url, nrows=50000)
        # Group by incident_id and take first event per incident
        if "number" in df.columns:
            incidents = df.groupby("number").first().reset_index()
            if len(incidents) > sample_size:
                incidents = incidents.sample(n=sample_size, random_state=42)
            print(f"[IT Ops] Loaded from UCI: {incidents.shape}")
            return incidents
    except Exception as e:
        print(f"[IT Ops] UCI download failed: {e}")

    print("[IT Ops] Generating synthetic IT Ops data...")
    return _generate_synthetic_it_ops(sample_size)


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
    """Fetch PM dataset from HuggingFace + synthetic augmentation."""
    print("[PM] Fetching Project Management dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("JohnVans123/ProjectManagement", split="train")
        df = ds.to_pandas()
        print(f"[PM] Loaded from HF: {df.shape}")
    except Exception as e:
        print(f"[PM] HF download failed: {e}, generating synthetic...")
        df = _generate_synthetic_pm(300)

    # Augment to target size
    if len(df) < augment_to:
        augmented = _augment_pm(df, target_size=augment_to)
        print(f"[PM] Augmented: {df.shape} -> {augmented.shape}")
        return augmented
    return df


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
    """Upload datasets to HuggingFace Hub."""
    try:
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi

        print("\nUploading to HuggingFace Hub...")

        # Raw datasets
        raw_dict = {}
        for domain, df in datasets.items():
            # Remove internal columns before upload
            upload_df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
            raw_dict[domain] = Dataset.from_pandas(upload_df)

        raw_ds = DatasetDict(raw_dict)
        raw_ds.push_to_hub("ricalanis/datasage-enterprise-raw")
        print("Uploaded: ricalanis/datasage-enterprise-raw")

        # Gold datasets
        gold_dict = {}
        for domain, df in gold_datasets.items():
            upload_df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
            gold_dict[domain] = Dataset.from_pandas(upload_df)

        gold_ds = DatasetDict(gold_dict)
        gold_ds.push_to_hub("ricalanis/datasage-enterprise-gold")
        print("Uploaded: ricalanis/datasage-enterprise-gold")

    except Exception as e:
        print(f"Upload failed: {e}")
        print("You can upload manually later with --upload flag after setting HF_TOKEN")


if __name__ == "__main__":
    main()
