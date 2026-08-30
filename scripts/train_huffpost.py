import pandas as pd
import argparse
import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import numpy as np
from collections import Counter
import time


def load_huffpost_data(data_path: str, min_samples: int = 200):
    """Load and preprocess HuffPost dataset"""
    df = pd.read_json(data_path, lines=True)

    # Combine headline and short_description as text
    headline = df["headline"].fillna("")
    short_desc = df["short_description"].fillna("")
    df["text"] = headline + ". " + short_desc

    # Clean category names
    df["category"] = df["category"].str.upper().str.replace(" ", "_")

    # Filter categories with enough samples
    category_counts = df["category"].value_counts()
    valid_categories = category_counts[category_counts >= min_samples].index
    df_filtered = df[df["category"].isin(valid_categories)]

    print(f"Original categories: {len(category_counts)}")
    print(f"Filtered categories (>={min_samples} samples): " f"{len(valid_categories)}")
    print(f"Total samples: {len(df_filtered)}")

    return df_filtered["text"].tolist(), df_filtered["category"].tolist()


def train_huffpost_baseline(
    data_path: str,
    output_dir: str,
    limit: int | None = None,
    min_samples: int = 200,
    max_features: int = 100000,
    ngram_max: int = 2,
    min_df: int = 2,
    stop_words_setting: str | None = "english",
):
    texts, labels = load_huffpost_data(data_path, min_samples=min_samples)

    if limit:
        texts = texts[:limit]
        labels = labels[:limit]

    # Show category distribution
    label_counts = Counter(labels)
    print("\nCategory distribution:")
    for cat, count in sorted(label_counts.items()):
        print(f"  {cat}: {count}")

    print("\nTrain/val split...")
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    print(
        "\nFitting TF-IDF vectorizer... "
        f"max_features={max_features}, ngram=(1,{ngram_max}), min_df={min_df}"
    )
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        lowercase=True,
        stop_words=stop_words_setting,
        min_df=min_df,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    print(
        "TF-IDF fit complete: "
        f"train_shape={X_train_vec.shape}, val_shape={X_val_vec.shape}, "
        f"elapsed={time.time() - t0:.1f}s"
    )

    # Use class weights for imbalanced data
    print("\nTraining Logistic Regression (saga, multinomial)...")
    t1 = time.time()
    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        multi_class="multinomial",
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train_vec, y_train)
    print(f"LR training complete: elapsed={time.time() - t1:.1f}s")

    preds = clf.predict(X_val_vec)
    macro_f1 = f1_score(y_val, preds, average="macro")
    weighted_f1 = f1_score(y_val, preds, average="weighted")

    print(f"\nMacro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(classification_report(y_val, preds, zero_division=0))

    # Save model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(clf, os.path.join(output_dir, "logreg.pkl"))

    # Create label mapping
    unique_labels = sorted(np.unique(labels))
    label_meta = {"classes_": np.array(unique_labels)}
    joblib.dump(label_meta, os.path.join(output_dir, "label_meta.pkl"))

    with open(
        os.path.join(output_dir, "metrics.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"macro_f1={macro_f1:.4f}\n")
        f.write(f"weighted_f1={weighted_f1:.4f}\n")
        f.write(f"categories={len(unique_labels)}\n")

    return macro_f1, weighted_f1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-path",
        required=True,
        help="Path to HuffPost JSON file",
    )
    ap.add_argument("--output-dir", default="models/classifier_huffpost")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--min-samples",
        type=int,
        default=200,
        help="Minimum samples per category",
    )
    ap.add_argument("--max-features", type=int, default=100000)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument(
        "--stop-words",
        type=str,
        default="english",
        help='Stop words setting ("english" or leave empty for None)',
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stop_words = args.stop_words if args.stop_words else None
    train_huffpost_baseline(
        args.data_path,
        args.output_dir,
        args.limit,
        min_samples=args.min_samples,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        stop_words_setting=stop_words,
    )
