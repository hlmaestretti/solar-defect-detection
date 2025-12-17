"""
Template drift detection script.

This script compares:
- a reference dataset (e.g. training data)
- a current dataset (e.g. recent predictions)

If drift_detected == True:
  -> trigger retraining pipeline
Else:
  -> do nothing
"""

import json
from pathlib import Path


def load_reference_data():
    """
    Load reference data used as a baseline.
    Expected to be implemented per project.
    """
    raise NotImplementedError


def load_current_data():
    """
    Load recent production data.
    Expected to be implemented per project.
    """
    raise NotImplementedError


def calculate_drift(reference_df, current_df):
    """
    Placeholder for drift calculation logic.
    Can be implemented using Evidently or custom logic.
    """
    drift_detected = False
    drift_score = 0.0

    return {
        "drift_detected": drift_detected,
        "drift_score": drift_score,
    }


def write_report(report, output_path="monitoring/drift_report.json"):
    Path(output_path).write_text(json.dumps(report, indent=2))


def main():
    reference_df = load_reference_data()
    current_df = load_current_data()

    report = calculate_drift(reference_df, current_df)
    write_report(report)


if __name__ == "__main__":
    main()
