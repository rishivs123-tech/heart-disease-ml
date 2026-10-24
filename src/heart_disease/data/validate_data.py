"""Dataset validation and Data Quality Report generator.

Matches PRD FR-01: Dataset validation acceptance criteria.
"""

import json
import logging
import os
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_COLS = [
    "dataset", "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal",
    "num", "target"
]

EXPECTED_ROW_COUNT = 920
EXPECTED_CLASSES = {0: 411, 1: 509}


class DatasetValidationError(Exception):
    """Raised when dataset fails fundamental acceptance criteria."""
    pass


def validate_dataset(df: pd.DataFrame) -> dict:
    """Validate dataset against PRD acceptance criteria and generate validation metrics."""
    report = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "record_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "missingness": {},
        "target_distribution": {},
        "source_distribution": {},
        "physiologically_invalid_zeros": {}
    }

    # 1. Check Row Count
    if len(df) != EXPECTED_ROW_COUNT:
        err = f"Expected {EXPECTED_ROW_COUNT} records, found {len(df)}."
        report["errors"].append(err)
        report["passed"] = False

    # 2. Check Expected Columns
    missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing_cols:
        err = f"Missing required columns: {missing_cols}"
        report["errors"].append(err)
        report["passed"] = False

    # 3. Target Distribution
    if "target" in df.columns:
        counts = df["target"].value_counts().to_dict()
        report["target_distribution"] = {int(k): int(v) for k, v in counts.items()}
        for cls, expected_cnt in EXPECTED_CLASSES.items():
            actual = report["target_distribution"].get(cls, 0)
            if actual != expected_cnt:
                warn = f"Target class {cls} count mismatch: expected {expected_cnt}, found {actual}."
                report["warnings"].append(warn)
    else:
        report["errors"].append("Target column 'target' not found in dataset.")
        report["passed"] = False

    # 4. Source Distribution
    if "dataset" in df.columns:
        report["source_distribution"] = {str(k): int(v) for k, v in df["dataset"].value_counts().items()}

    # 5. Missingness Audit
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / len(df)) * 100, 2)
        report["missingness"][col] = {
            "null_count": null_count,
            "null_pct": null_pct
        }

    # 6. Check for remaining physiologically invalid zeros in numeric columns
    for col in ["trestbps", "chol"]:
        if col in df.columns:
            zero_count = int((df[col] == 0).sum())
            report["physiologically_invalid_zeros"][col] = zero_count
            if zero_count > 0:
                report["warnings"].append(f"Column '{col}' still contains {zero_count} zero values.")

    return report


def generate_data_quality_report(
    data_path: str = "data/processed/heart_disease_cleaned.csv",
    output_dir: str = "results/logs"
) -> dict:
    """Generate both JSON and Markdown Data Quality Reports."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Run load_data.py first.")

    df = pd.read_csv(data_path)
    report = validate_dataset(df)

    # Save JSON report
    json_path = os.path.join(output_dir, "data_quality_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save Markdown report
    md_path = os.path.join(output_dir, "data_quality_report.md")
    md_content = f"""# Data Quality and Acceptance Report
**Generated for**: 920-Record UCI Heart Disease Cohort  
**Validation Status**: {'✅ PASSED' if report['passed'] else '❌ FAILED'}  
**Total Records**: {report['record_count']} (Expected: {EXPECTED_ROW_COUNT})  
**Total Columns**: {report['column_count']}  

---

## 1. Class Distribution
- **Class 0 (Disease Absent)**: {report['target_distribution'].get(0, 0)} ({round(report['target_distribution'].get(0, 0)/report['record_count']*100, 2)}%)
- **Class 1 (Disease Present)**: {report['target_distribution'].get(1, 0)} ({round(report['target_distribution'].get(1, 0)/report['record_count']*100, 2)}%)

## 2. Institutional Cohort Breakdown
"""
    for site, count in report["source_distribution"].items():
        md_content += f"- **{site}**: {count} records ({round(count/report['record_count']*100, 2)}%)\n"

    md_content += """
## 3. Missingness Summary per Feature

| Feature | Missing Values (Count) | Missing Percentage (%) |
|---|---|---|
"""
    for col, mdata in report["missingness"].items():
        md_content += f"| `{col}` | {mdata['null_count']} | {mdata['null_pct']}% |\n"

    md_content += "\n## 4. Validation Checks & Warnings\n"
    if report["errors"]:
        md_content += "### ❌ Errors:\n"
        for err in report["errors"]:
            md_content += f"- {err}\n"
    else:
        md_content += "- All fundamental schema and record count assertions passed successfully.\n"

    if report["warnings"]:
        md_content += "### ⚠️ Warnings / Notes:\n"
        for w in report["warnings"]:
            md_content += f"- {w}\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Data Quality Report saved to %s and %s", json_path, md_path)
    if not report["passed"]:
        raise DatasetValidationError(f"Dataset validation failed: {report['errors']}")

    return report


if __name__ == "__main__":
    generate_data_quality_report()
