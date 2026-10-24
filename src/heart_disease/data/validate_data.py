"""Dataset validation and quality check utilities."""

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
    """Raised when dataset fails basic schema or count checks."""
    pass


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run data integrity and schema validation checks."""
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

    # row count check
    if len(df) != EXPECTED_ROW_COUNT:
        err = f"Expected {EXPECTED_ROW_COUNT} records, found {len(df)}."
        report["errors"].append(err)
        report["passed"] = False

    # column check
    missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing_cols:
        err = f"Missing required columns: {missing_cols}"
        report["errors"].append(err)
        report["passed"] = False

    # target distribution check
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

    # source site distribution
    if "dataset" in df.columns:
        report["source_distribution"] = {str(k): int(v) for k, v in df["dataset"].value_counts().items()}

    # missingness calculation
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / len(df)) * 100, 2)
        report["missingness"][col] = {
            "null_count": null_count,
            "null_pct": null_pct
        }

    # check remaining zeros in continuous features
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
    """Generate JSON and Markdown data quality summary reports."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Run load_data.py first.")

    df = pd.read_csv(data_path)
    report = validate_dataset(df)

    json_path = os.path.join(output_dir, "data_quality_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = os.path.join(output_dir, "data_quality_report.md")
    status_str = "PASSED" if report["passed"] else "FAILED"
    md_content = f"""# Data Quality and Validation Report
**Dataset**: Combined UCI Heart Disease Cohort  
**Status**: {status_str}  
**Total Records**: {report['record_count']}  
**Columns**: {report['column_count']}  

---

## 1. Class Distribution
- **Class 0 (No Disease)**: {report['target_distribution'].get(0, 0)} ({round(report['target_distribution'].get(0, 0)/report['record_count']*100, 2)}%)
- **Class 1 (Disease Present)**: {report['target_distribution'].get(1, 0)} ({round(report['target_distribution'].get(1, 0)/report['record_count']*100, 2)}%)

## 2. Records by Source Center
"""
    for site, count in report["source_distribution"].items():
        md_content += f"- **{site}**: {count} ({round(count/report['record_count']*100, 2)}%)\n"

    md_content += """
## 3. Missing Value Summary

| Feature | Missing Count | Missing (%) |
|---|---|---|
"""
    for col, mdata in report["missingness"].items():
        md_content += f"| `{col}` | {mdata['null_count']} | {mdata['null_pct']}% |\n"

    md_content += "\n## 4. Validation Checks\n"
    if report["errors"]:
        md_content += "### Errors:\n"
        for err in report["errors"]:
            md_content += f"- {err}\n"
    else:
        md_content += "- All schema and record count checks passed.\n"

    if report["warnings"]:
        md_content += "### Warnings / Notes:\n"
        for w in report["warnings"]:
            md_content += f"- {w}\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Data report written to %s and %s", json_path, md_path)
    if not report["passed"]:
        raise DatasetValidationError(f"Validation failed: {report['errors']}")

    return report


if __name__ == "__main__":
    generate_data_quality_report()
