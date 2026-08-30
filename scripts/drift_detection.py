"""Drift detection utilities using Evidently."""

import os
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DriftDetector:
    """Monitor data drift in news classification."""

    def __init__(self, reference_data_path: Optional[str] = None):
        self.reference_data_path = reference_data_path or "data/reference_dataset.csv"
        self.reference_data = None
        self._load_reference_data()

    def _load_reference_data(self):
        """Load reference dataset for drift comparison."""
        if Path(self.reference_data_path).exists():
            self.reference_data = pd.read_csv(self.reference_data_path)
            logger.info(
                "Loaded reference dataset with %d samples", len(self.reference_data)
            )
        else:
            logger.warning(
                "Reference dataset not found at %s", self.reference_data_path
            )

    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.

        Args:
            current_data: DataFrame with current predictions/data

        Returns:
            Dictionary with drift analysis results
        """
        if self.reference_data is None:
            return {"error": "No reference data available"}

        try:
            # Simple drift detection based on category distribution
            ref_categories = self.reference_data["category"].value_counts(
                normalize=True
            )
            curr_categories = current_data["category"].value_counts(normalize=True)

            # Calculate distribution difference
            all_categories = set(ref_categories.index) | set(curr_categories.index)
            drift_score = 0.0
            drifted_features = 0

            for cat in all_categories:
                ref_prob = ref_categories.get(cat, 0.0)
                curr_prob = curr_categories.get(cat, 0.0)
                diff = abs(ref_prob - curr_prob)
                drift_score += diff

                # Count as drifted if difference > 0.1
                if diff > 0.1:
                    drifted_features += 1

            # Normalize drift score
            drift_score = min(drift_score, 2.0)  # Cap at 2.0

            # Dataset drift if score > 0.5 or significant category changes
            dataset_drift = (
                drift_score > 0.5 or drifted_features > len(all_categories) * 0.3
            )

            return {
                "dataset_drift": dataset_drift,
                "drift_score": drift_score,
                "drifted_features": drifted_features,
                "total_features": len(all_categories),
                "drift_share": drifted_features / len(all_categories),
                "reference_distribution": ref_categories.to_dict(),
                "current_distribution": curr_categories.to_dict(),
            }

        except Exception as e:
            logger.error("Error detecting drift: %s", e)
            return {"error": str(e)}

    def update_reference_data(self, new_data: pd.DataFrame):
        """Update reference dataset with new data."""
        try:
            # Save current reference as backup
            if self.reference_data is not None:
                backup_path = f"{self.reference_data_path}.backup"
                self.reference_data.to_csv(backup_path, index=False)

            # Update reference data
            self.reference_data = new_data.copy()
            self.reference_data.to_csv(self.reference_data_path, index=False)
            logger.info("Updated reference dataset with %d samples", len(new_data))

        except Exception as e:
            logger.error("Error updating reference data: %s", e)

    def get_drift_report(self, current_data: pd.DataFrame) -> str:
        """Generate a human-readable drift report."""
        drift_results = self.detect_drift(current_data)

        if "error" in drift_results:
            return f"Drift detection failed: {drift_results['error']}"

        dataset_drift = drift_results.get("dataset_drift", False)
        drift_score = drift_results.get("drift_score", 0.0)
        drifted_features = drift_results.get("drifted_features", 0)
        total_features = drift_results.get("total_features", 0)

        report = f"""
Data Drift Report:
==================
Dataset Drift Detected: {'YES' if dataset_drift else 'NO'}
Drift Score: {drift_score:.3f}
Drifted Categories: {drifted_features}/{total_features}
"""

        if dataset_drift:
            report += "\n⚠️  ACTION REQUIRED: Consider model retraining\n"
        else:
            report += "\n✅ Data distribution appears stable\n"

        return report


# Global drift detector instance
_drift_detector = None


def get_drift_detector() -> DriftDetector:
    """Get global drift detector instance."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector


def check_drift_for_predictions(predictions_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Check drift for prediction results.

    Args:
        predictions_data: DataFrame with prediction data

    Returns:
        Drift analysis results
    """
    detector = get_drift_detector()
    return detector.detect_drift(predictions_data)


def generate_reference_dataset():
    """Generate initial reference dataset from training data."""
    try:
        # Create a simple reference dataset
        data = {
            "text": [
                "Apple announces new iPhone with advanced features",
                "Tesla stock rises after earnings report",
                "New COVID variant detected in laboratory",
                "Federal Reserve announces interest rate decision",
                "Climate change summit reaches new agreements",
                "SpaceX successfully launches satellite mission",
                "New movie breaks box office records",
                "Scientists discover new species in Amazon",
                "Olympic games begin with opening ceremony",
                "Tech companies face antitrust investigations",
            ],
            "category": [
                "TECH",
                "BUSINESS",
                "HEALTH",
                "BUSINESS",
                "ENVIRONMENT",
                "SCIENCE",
                "ENTERTAINMENT",
                "SCIENCE",
                "SPORTS",
                "BUSINESS",
            ],
        }

        df = pd.DataFrame(data)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/reference_dataset.csv", index=False)
        print(f"Created reference dataset with {len(df)} samples")

    except Exception as e:
        print(f"Error creating reference dataset: {e}")


if __name__ == "__main__":
    # Generate reference dataset if it doesn't exist
    if not Path("data/reference_dataset.csv").exists():
        generate_reference_dataset()

    # Test drift detection
    detector = get_drift_detector()
    if detector.reference_data is not None:
        # Create some test data
        test_data = pd.DataFrame(
            {
                "text": [
                    "Apple announces new iPhone",
                    "Tesla stock rises",
                    "New COVID variant detected",
                ],
                "category": ["TECH", "BUSINESS", "HEALTH"],
            }
        )

        report = detector.get_drift_report(test_data)
        print(report)
