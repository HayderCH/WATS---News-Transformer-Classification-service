import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def prepare_eval_split(
    data_path: str,
    label2id: dict[str, int],
    use_headline_only: bool = False,
    test_size: float = 0.15,
    seed: int = 42,
    limit: int | None = None,
):
    # nosec B615: local JSON file, not a remote download
    ds: Dataset = load_dataset(
        "json",
        data_files=data_path,
        split="train",
    )  # nosec B615

    def build_fields(example):
        headline = example.get("headline") or ""
        short_desc = example.get("short_description") or ""
        category = (example.get("category") or "").upper().replace(" ", "_")
        text = (
            headline.strip()
            if use_headline_only
            else (headline + ". " + short_desc).strip()
        )
        return {"text": text, "category": category}

    keep_cols = [
        c
        for c in ds.column_names
        if c in {"headline", "short_description", "category"}
    ]
    ds = ds.map(
        build_fields,
        remove_columns=[c for c in ds.column_names if c not in keep_cols],
    )

    # Keep only categories seen during training
    def has_label(x):
        return x["category"] in label2id

    ds = ds.filter(has_label)
    ds = ds.map(lambda x: {"labels": label2id[x["category"]]})

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # Stratified split by integer labels
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(ds["labels"]):
        by_label[int(lbl)].append(idx)

    val_idx: list[int] = []
    for _, idxs in by_label.items():
        n_val = max(1, int(round(len(idxs) * test_size)))
        choice = rng.choice(idxs, size=n_val, replace=False)
        val_idx.extend(choice.tolist())

    val_idx = sorted(set(val_idx))
    all_idx = set(range(len(ds)))
    train_idx = sorted(all_idx - set(val_idx))

    train_ds = ds.select(train_idx)
    val_ds = ds.select(val_idx)

    return train_ds, val_ds


def evaluate(
    model_dir: str,
    data_path: str,
    batch_size: int = 64,
    max_length: int = 256,
    use_headline_only: bool = False,
    test_size: float = 0.15,
    seed: int = 42,
    limit: int | None = None,
):
    model_root = Path(model_dir)

    # Locate label metadata and model checkpoint
    meta_path = model_root / "label_meta.json"
    if not meta_path.exists():
        # Try parent if user passed the 'best' subdir
        if (model_root.parent / "label_meta.json").exists():
            meta_path = model_root.parent / "label_meta.json"
            model_root = model_root.parent
        else:
            raise FileNotFoundError(
                f"label_meta.json not found near {model_dir}"
            )

    with open(meta_path, "r") as f:
        label_meta = json.load(f)

    id2label = {int(k): v for k, v in label_meta["id2label"].items()}
    label2id = {v: int(k) for k, v in id2label.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tdir = model_root / "best"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tdir),
        local_files_only=True,
    )  # nosec B615
    model = AutoModelForSequenceClassification.from_pretrained(
        str(tdir),
        local_files_only=True,
    ).to(
        device
    )  # nosec B615
    model.eval()

    _, val_ds = prepare_eval_split(
        data_path=data_path,
        label2id=label2id,
        use_headline_only=use_headline_only,
        test_size=test_size,
        seed=seed,
        limit=limit,
    )

    # Tokenize
    def tok_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    cols_to_remove = [c for c in val_ds.column_names if c != "labels"]
    val_tok = val_ds.map(tok_fn, batched=True, remove_columns=cols_to_remove)
    val_tok.set_format(type="torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(val_tok, batch_size=batch_size, collate_fn=collator)

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").numpy().tolist()
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k in {"input_ids", "attention_mask", "token_type_ids"}
            }
            outputs = model(**inputs)
            logits = outputs.logits
            preds = (
                torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            )
            all_labels.extend(labels)
            all_preds.extend(preds)

    acc = accuracy_score(all_labels, all_preds)
    f1_m = f1_score(all_labels, all_preds, average="macro")
    f1_w = f1_score(all_labels, all_preds, average="weighted")

    print({"accuracy": acc, "f1_macro": f1_m, "f1_weighted": f1_w})

    report = classification_report(
        all_labels,
        all_preds,
        target_names=[id2label[i] for i in sorted(id2label)],
    )
    cm = confusion_matrix(all_labels, all_preds)

    out_path = model_root / "eval_report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"accuracy={acc:.4f}\n")
        f.write(f"f1_macro={f1_m:.4f}\n")
        f.write(f"f1_weighted={f1_w:.4f}\n\n")
        f.write("Classification Report\n")
        f.write(report)
        f.write("\n\nConfusion Matrix (rows=true, cols=pred)\n")
        # Write a compact CM
        for row in cm:
            f.write(",".join(str(int(x)) for x in row) + "\n")

    print(f"Saved detailed report to: {out_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Path to trained model dir",
    )
    ap.add_argument(
        "--data-path",
        required=True,
        help="Path to HuffPost JSON data",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--use-headline-only", action="store_true")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    evaluate(
        model_dir=a.model_dir,
        data_path=a.data_path,
        batch_size=a.batch_size,
        max_length=a.max_length,
        use_headline_only=a.use_headline_only,
        test_size=a.test_size,
        seed=a.seed,
        limit=a.limit,
    )
