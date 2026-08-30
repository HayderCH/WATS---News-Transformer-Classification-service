#!/usr/bin/env python3
"""
Data Quality Checks for News Topic Classification Dataset
"""

import json
import pandas as pd
from collections import Counter
import argparse


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the JSON dataset into a DataFrame."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def check_duplicates(df: pd.DataFrame) -> dict:
    """Check for duplicate entries."""
    total = len(df)
    duplicates = df.duplicated().sum()
    text_duplicates = df.duplicated(subset=["headline", "short_description"]).sum()
    return {
        "total_rows": total,
        "duplicate_rows": duplicates,
        "duplicate_texts": text_duplicates,
        "duplicate_percentage": (duplicates / total) * 100 if total > 0 else 0,
    }


def check_class_balance(df: pd.DataFrame) -> dict:
    """Check class distribution."""
    category_counts = Counter(df["category"])
    total = len(df)
    balance = {
        cat: {"count": count, "percentage": (count / total) * 100}
        for cat, count in category_counts.items()
    }
    return {
        "total_samples": total,
        "num_classes": len(category_counts),
        "class_distribution": balance,
        "most_common": category_counts.most_common(5),
        "least_common": category_counts.most_common()[-5:],
    }


def check_text_lengths(df: pd.DataFrame) -> dict:
    """Check text length statistics."""
    df["headline_len"] = df["headline"].str.len()
    df["desc_len"] = df["short_description"].str.len()
    return {
        "headline_stats": df["headline_len"].describe().to_dict(),
        "description_stats": df["desc_len"].describe().to_dict(),
        "empty_headlines": (df["headline_len"] == 0).sum(),
        "empty_descriptions": (df["desc_len"] == 0).sum(),
    }


def main(filepath: str):
    """Run all quality checks."""
    print(f"Loading dataset from {filepath}...")
    df = load_dataset(filepath)

    print("\n=== DUPLICATE CHECK ===")
    dup_results = check_duplicates(df)
    for key, value in dup_results.items():
        print(f"{key}: {value}")

    print("\n=== CLASS BALANCE CHECK ===")
    balance_results = check_class_balance(df)
    for key, value in balance_results.items():
        if key == "class_distribution":
            print(f"{key}:")
            for cat, stats in value.items():
                print(f"  {cat}: {stats}")
        else:
            print(f"{key}: {value}")

    print("\n=== TEXT LENGTH CHECK ===")
    length_results = check_text_lengths(df)
    for key, value in length_results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Quality Checks")
    parser.add_argument("filepath", help="Path to the dataset JSON file")
    args = parser.parse_args()
    main(args.filepath)
