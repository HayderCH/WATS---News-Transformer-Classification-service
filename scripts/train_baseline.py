import argparse
import os
import joblib
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
import numpy as np

from app.core.config import get_settings
from app.services import mlflow_run

HF_DATASET_AG_NEWS_REVISION = os.getenv(
    "HF_DATASET_AG_NEWS_REVISION",
    "refs/convert/parquet",
)


def load_ag_news(limit: int | None = None):
    ds = load_dataset(
        "ag_news",
        revision=HF_DATASET_AG_NEWS_REVISION,
    )  # nosec B615
    texts = []
    labels = []
    for row in ds["train"]:
        texts.append(row["text"])
        labels.append(row["label"])
    if limit:
        texts = texts[:limit]
        labels = labels[:limit]
    return texts, labels


def load_huffpost(limit: int | None = None):
    import json

    texts = []
    labels = []
    label_to_idx = {}
    idx = 0
    with open(
        "data/raw/huffpost/News_Category_Dataset_v3.json", "r", encoding="utf-8"
    ) as f:
        for line in f:
            row = json.loads(line)
            text = f"{row['headline']}. {row['short_description']}"
            label = row["category"]
            if label not in label_to_idx:
                label_to_idx[label] = idx
                idx += 1
            texts.append(text)
            labels.append(label_to_idx[label])
            if limit and len(texts) >= limit:
                break
    return texts, labels


def train_baseline(
    output_dir: str, limit: int | None = None, dataset: str = "huffpost"
):
    settings = get_settings()
    tags = {
        "model": "baseline",
        "classifier_version": settings.classifier_version,
        "classifier_backend": settings.classifier_backend,
    }

    with mlflow_run("baseline", tags=tags) as mlflow_ctx:
        if dataset == "ag_news":
            texts, labels = load_ag_news(limit)
        elif dataset == "huffpost":
            texts, labels = load_huffpost(limit)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.15, random_state=42, stratify=labels
        )
        vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            lowercase=True,
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)
        clf = LogisticRegression(
            max_iter=1000,
            n_jobs=-1 if hasattr(LogisticRegression, "n_jobs") else None,
        )
        clf.fit(X_train_vec, y_train)
        train_iterations = None
        if hasattr(clf, "n_iter_"):
            train_iterations = int(np.max(clf.n_iter_))
        preds = clf.predict(X_val_vec)
        macro_f1 = f1_score(y_val, preds, average="macro")
        accuracy = accuracy_score(y_val, preds)
        report = classification_report(y_val, preds)

        print(f"Macro F1: {macro_f1:.4f}")
        print(report)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(
            vectorizer,
            os.path.join(output_dir, "tfidf_vectorizer.pkl"),
        )
        joblib.dump(clf, os.path.join(output_dir, "logreg.pkl"))
        # save label encoder metadata
        joblib.dump(
            {"classes_": np.unique(labels)},
            os.path.join(output_dir, "label_meta.pkl"),
        )
        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"macro_f1={macro_f1:.4f}\n")
            f.write(f"accuracy={accuracy:.4f}\n")
            f.write(f"train_samples={len(X_train)}\n")
            f.write(f"val_samples={len(X_val)}\n")
        with open(
            os.path.join(output_dir, "classification_report.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report)

        if mlflow_ctx:
            mlflow_ctx.log_params(
                {
                    "limit": str(limit) if limit is not None else "full",
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "vectorizer_max_features": vectorizer.max_features,
                    "vectorizer_ngram_range": str(vectorizer.ngram_range),
                    "classifier_max_iter": clf.max_iter,
                    "solver": getattr(clf, "solver", "lbfgs"),
                }
            )
            mlflow_ctx.log_metric("macro_f1", macro_f1)
            mlflow_ctx.log_metric("accuracy", accuracy)
            if train_iterations is not None:
                mlflow_ctx.log_metric(
                    "classifier_iterations",
                    float(train_iterations),
                )
            mlflow_ctx.log_artifacts(output_dir)

        return macro_f1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="models/classifier")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit training samples for speed",
    )
    ap.add_argument(
        "--dataset",
        default="huffpost",
        choices=["ag_news", "huffpost"],
        help="Dataset to train on",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_baseline(args.output_dir, args.limit, args.dataset)
