import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from transformers import DebertaV2Tokenizer
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn


HF_DATASET_JSON_REVISION = os.getenv("HF_DATASET_JSON_REVISION", "main")
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "main")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def prepare_huffpost_dataset(
    data_path: str,
    min_samples: int = 200,
    limit: int | None = None,
    test_size: float = 0.15,
    seed: int = 42,
    use_headline_only: bool = False,
    extra_examples: list[dict[str, str]] | None = None,
):
    # Load JSONL via HF datasets
    ds: Dataset = load_dataset(
        "json",
        data_files=data_path,
        split="train",
        revision=HF_DATASET_JSON_REVISION,
    )  # nosec B615

    if extra_examples:
        normalized = []
        for row in extra_examples:
            normalized.append(
                {
                    "headline": row.get("headline", ""),
                    "short_description": row.get("short_description", ""),
                    "category": row.get("category", ""),
                }
            )
        if normalized:
            extra_ds = Dataset.from_list(normalized)
            ds = concatenate_datasets([ds, extra_ds])

    # Build text (headline + short_description) and normalize category
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

    ds = ds.map(
        build_fields,
        remove_columns=[
            c
            for c in ds.column_names
            if c not in {"headline", "short_description", "category"}
        ],
    )

    # Filter categories with at least min_samples
    counts = Counter(ds["category"])
    valid = {c for c, n in counts.items() if n >= min_samples}
    ds = ds.filter(lambda x: x["category"] in valid)

    # Optionally limit for quicker runs
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    # Make label mappings
    label_names = sorted(sorted(valid))  # stable order
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}

    # Add integer label column as 'labels' (Trainer expects this key)
    ds = ds.map(lambda x: {"labels": label2id[x["category"]]})

    # Stratified train/val split by label to preserve class ratios
    rng = np.random.default_rng(seed)
    indices_by_label: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(ds["labels"]):
        indices_by_label[int(lbl)].append(idx)

    val_indices: list[int] = []
    for lbl, idxs in indices_by_label.items():
        n_val = max(1, int(round(len(idxs) * test_size)))
        choice = rng.choice(idxs, size=n_val, replace=False)
        val_indices.extend(choice.tolist())
    val_indices = sorted(set(val_indices))
    all_indices = set(range(len(ds)))
    train_indices = sorted(all_indices - set(val_indices))

    train_ds = ds.select(train_indices)
    val_ds = ds.select(val_indices)

    # Simple dataset stats
    print(
        "Samples after filtering: "
        f"{len(ds)} across {len(label_names)} categories"
    )

    return train_ds, val_ds, label_names, label2id, id2label


def train_huffpost_transformer(
    data_path: str,
    model_name: str = "microsoft/deberta-v3-base",
    output_dir: str = "models/transformer_huffpost",
    min_samples: int = 200,
    limit: int | None = None,
    epochs: int = 3,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    grad_accum: int = 1,
    max_length: int = 256,
    learning_rate: float = 2e-5,
    fallback_model_name: str = "roberta-base",
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    scheduler: str = "linear",
    early_stopping_patience: int = 3,
    eval_strategy: str = "epoch",
    eval_steps: int = 1000,
    save_total_limit: int = 2,
    label_smoothing: float = 0.1,
    dataloader_num_workers: int = 4,
    use_headline_only: bool = False,
    test_size: float = 0.15,
    seed: int = 42,
    gradient_checkpointing: bool = True,
    use_class_weights: bool = True,
    extra_examples: list[dict[str, str]] | None = None,
):
    (
        train_ds,
        val_ds,
        label_names,
        label2id,
        id2label,
    ) = prepare_huffpost_dataset(
        data_path,
        min_samples=min_samples,
        limit=limit,
        test_size=test_size,
        seed=seed,
        use_headline_only=use_headline_only,
        extra_examples=extra_examples,
    )

    if extra_examples:
        print(
            "Augmented training set with "
            f"{len(extra_examples)} human-labeled items"
        )

    # Load tokenizer with robust fallbacks
    use_model_name = model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            use_model_name,
            use_fast=True,
            revision=HF_MODEL_REVISION,
        )  # nosec B615
    except Exception as e_fast:
        print(
            "Fast tokenizer failed, trying explicit slow DeBERTa tokenizer:",
            e_fast,
        )
        try:
            tokenizer = DebertaV2Tokenizer.from_pretrained(
                use_model_name,
                revision=HF_MODEL_REVISION,
            )  # nosec B615
        except Exception as e_slow:
            print(
                "DeBERTa tokenizer failed in both fast and slow modes. "
                f"Falling back to {fallback_model_name}. Details:",
                e_slow,
            )
            use_model_name = fallback_model_name
            tokenizer = AutoTokenizer.from_pretrained(
                use_model_name,
                use_fast=True,
                revision=HF_MODEL_REVISION,
            )  # nosec B615

    model = AutoModelForSequenceClassification.from_pretrained(
        use_model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        revision=HF_MODEL_REVISION,
    )  # nosec B615

    # Enable faster matmul; RTX 40-series supports bf16
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.set_float32_matmul_precision("high")
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    use_bf16 = bool(
        use_cuda and callable(is_bf16_supported) and is_bf16_supported()
    )
    use_fp16 = bool(use_cuda and not use_bf16)

    # Diagnostics: show GPU and precision selections
    print(
        "CUDA available:",
        use_cuda,
        "| device_count=",
        torch.cuda.device_count(),
    )
    if use_cuda and torch.cuda.device_count() > 0:
        try:
            names = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
            print("GPU(s):", ", ".join(names))
        except Exception:
            pass
    print("Precision:", "bf16=" + str(use_bf16) + ", fp16=" + str(use_fp16))

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    # Keep the 'labels' column so the model computes loss
    cols_to_remove_train = [c for c in train_ds.column_names if c != "labels"]
    cols_to_remove_val = [c for c in val_ds.column_names if c != "labels"]
    train_tok = train_ds.map(
        tokenize_fn, batched=True, remove_columns=cols_to_remove_train
    )
    val_tok = val_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=cols_to_remove_val,
    )

    # Build dynamic training arguments
    args_kwargs = dict(
        output_dir=f"{output_dir}/checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type=scheduler,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=save_total_limit,
        dataloader_pin_memory=use_cuda,
        dataloader_num_workers=dataloader_num_workers,
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
        optim=("adamw_torch_fused" if use_cuda else "adamw_torch"),
        gradient_checkpointing=gradient_checkpointing,
        label_smoothing_factor=label_smoothing,
        save_safetensors=True,
        seed=seed,
    )

    if eval_strategy == "steps":
        args_kwargs.update(
            dict(
                evaluation_strategy="steps",
                eval_steps=eval_steps,
                save_strategy="steps",
                save_steps=eval_steps,
            )
        )
    else:
        args_kwargs.update(
            dict(
                evaluation_strategy="epoch",
                save_strategy="epoch",
            )
        )

    args = TrainingArguments(**args_kwargs)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Optional class-weighted loss to improve macro-F1 on imbalanced labels
    trainer_cls = Trainer
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            )
        ],
    )

    if use_class_weights:
        counts = Counter(train_ds["labels"])  # type: ignore[index]
        num_labels = len(label_names)
        total = len(train_ds)
        weights = np.zeros(num_labels, dtype=np.float32)
        for i in range(num_labels):
            c = counts.get(i, 1)
            weights[i] = total / (num_labels * max(1, c))
        class_weights = torch.tensor(weights, dtype=torch.float)

        class WeightedTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                self.class_weights = kwargs.pop("class_weights")
                super().__init__(*args, **kwargs)

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                # Ensure stable dtype for loss computation
                logits = logits.float()
                loss_fct = nn.CrossEntropyLoss(
                    weight=self.class_weights.to(logits.device),
                    label_smoothing=getattr(
                        self.args,
                        "label_smoothing_factor",
                        0.0,
                    ),
                )
                loss = loss_fct(
                    logits.view(-1, self.model.config.num_labels),
                    labels.view(-1),
                )
                return (loss, outputs) if return_outputs else loss

        trainer_cls = WeightedTrainer
        trainer_kwargs["class_weights"] = class_weights

    trainer = trainer_cls(**trainer_kwargs)

    # Trainer device info
    try:
        print("Trainer device:", trainer.args.device)
    except Exception:
        pass

    print("Starting HuffPost transformer training…")
    trainer.train()

    results = trainer.evaluate()
    print(f"Final results: {results}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(f"{output_dir}/best")
    tokenizer.save_pretrained(f"{output_dir}/best")

    label_meta = {
        "label_names": label_names,
        "id2label": id2label,
        "label2id": label2id,
    }
    with open(f"{output_dir}/label_meta.json", "w") as f:
        json.dump(label_meta, f, indent=2)

    with open(f"{output_dir}/metrics.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}={v:.4f}\n")

    return results


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-path",
        required=True,
        help="Path to News_Category_*.json",
    )
    ap.add_argument("--model-name", default="microsoft/deberta-v3-base")
    ap.add_argument("--output-dir", default="models/transformer_huffpost")
    ap.add_argument("--min-samples", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--fallback-model-name", type=str, default="roberta-base")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
        ],
    )
    ap.add_argument("--early-stopping-patience", type=int, default=3)
    ap.add_argument(
        "--eval-strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps"],
    )
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--dataloader-num-workers", type=int, default=4)
    ap.add_argument("--use-headline-only", action="store_true")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--no-class-weights", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    train_huffpost_transformer(
        data_path=a.data_path,
        model_name=a.model_name,
        output_dir=a.output_dir,
        min_samples=a.min_samples,
        limit=a.limit,
        epochs=a.epochs,
        train_batch_size=a.train_batch_size,
        eval_batch_size=a.eval_batch_size,
        grad_accum=a.grad_accum,
        max_length=a.max_length,
        learning_rate=a.learning_rate,
        fallback_model_name=a.fallback_model_name,
        warmup_ratio=a.warmup_ratio,
        weight_decay=a.weight_decay,
        scheduler=a.scheduler,
        early_stopping_patience=a.early_stopping_patience,
        eval_strategy=a.eval_strategy,
        eval_steps=a.eval_steps,
        save_total_limit=a.save_total_limit,
        label_smoothing=a.label_smoothing,
        dataloader_num_workers=a.dataloader_num_workers,
        use_headline_only=a.use_headline_only,
        test_size=a.test_size,
        seed=a.seed,
        gradient_checkpointing=(not a.no_gradient_checkpointing),
        use_class_weights=(not a.no_class_weights),
    )
