#!/usr/bin/env python3
"""
Advanced evaluation script with F1 per class and bias detection.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import pandas as pd
import typer


def load_data(filepath: str) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def evaluate_model(
    model_predictions: list[dict], true_labels: list[str], label_names: list[str]
):
    """Evaluate model with advanced metrics."""
    pred_labels = [pred["categories"][0]["name"] for pred in model_predictions]

    # Classification report
    report = classification_report(
        true_labels, pred_labels, target_names=label_names, output_dict=True
    )
    print("Classification Report:")
    print(classification_report(true_labels, pred_labels, target_names=label_names))

    # F1 per class
    f1_per_class = {label: report[label]["f1-score"] for label in label_names}
    print("\nF1 per Class:")
    for label, f1 in f1_per_class.items():
        print(f"{label}: {f1:.3f}")

    # Overall F1
    overall_f1 = f1_score(true_labels, pred_labels, average="weighted")
    print(f"\nOverall Weighted F1: {overall_f1:.3f}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=label_names)
    print("\nConfusion Matrix:")
    print(cm)

    # Bias detection: Check if certain classes are underperforming
    avg_f1 = np.mean(list(f1_per_class.values()))
    biased_classes = [label for label, f1 in f1_per_class.items() if f1 < avg_f1 * 0.8]
    if biased_classes:
        print(f"\nPotentially Biased Classes (F1 < 80% of average): {biased_classes}")
    else:
        print("\nNo significant bias detected.")

    return {
        "report": report,
        "f1_per_class": f1_per_class,
        "overall_f1": overall_f1,
        "confusion_matrix": cm.tolist(),
        "biased_classes": biased_classes,
    }


def main(
    predictions_file: str = typer.Argument(
        ..., help="JSON file with model predictions"
    ),
    true_labels_file: str = typer.Argument(..., help="JSON file with true labels"),
    labels_file: str = typer.Option(None, help="File with label names, one per line"),
):
    # Load predictions (assume format from API)
    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    # Load true labels
    true_df = load_data(true_labels_file)
    true_labels = true_df["category"].tolist()

    # Load label names
    if labels_file:
        with open(labels_file, "r") as f:
            label_names = [line.strip() for line in f]
    else:
        label_names = sorted(set(true_labels))

    evaluate_model(predictions, true_labels, label_names)


if __name__ == "__main__":
    typer.run(main)
