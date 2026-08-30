import argparse
import json
import os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from app.core.config import get_settings
from app.services import mlflow_run

HF_DATASET_AG_NEWS_REVISION = os.getenv(
    "HF_DATASET_AG_NEWS_REVISION",
    "refs/convert/parquet",
)
HF_MODEL_REVISION = os.getenv("HF_MODEL_REVISION", "main")


def load_ag_news_for_transformer():
    """Load AG News and format for transformers"""
    ds = load_dataset(
        "ag_news",
        revision=HF_DATASET_AG_NEWS_REVISION,
    )  # nosec B615

    # Map labels to text names for better training
    label_names = ["WORLD", "SPORTS", "BUSINESS", "SCIENCE_TECH"]

    def add_label_names(example):
        example["label_name"] = label_names[example["label"]]
        return example

    train_ds = ds["train"].map(add_label_names)
    test_ds = ds["test"].map(add_label_names)

    return train_ds, test_ds, label_names


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


def train_transformer_classifier(
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "models/transformer_agnews",
    limit: int = None,
    epochs: int = 3,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    grad_accum: int = 1,
):
    train_ds, test_ds, label_names = load_ag_news_for_transformer()

    if limit:
        train_ds = train_ds.select(range(min(limit, len(train_ds))))
        test_ds = test_ds.select(range(min(limit // 4, len(test_ds))))

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=HF_MODEL_REVISION,
    )  # nosec B615
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        revision=HF_MODEL_REVISION,
    )  # nosec B615

    # Optional GPU niceties
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    def tokenize_function(examples):
        # Let the data collator handle dynamic padding;
        # keep truncation for safety
        return tokenizer(examples["text"], truncation=True, max_length=512)

    # Tokenize datasets
    train_tokenized = train_ds.map(tokenize_function, batched=True)
    test_tokenized = test_ds.map(tokenize_function, batched=True)

    # Dynamically set warmup based on dataset size if a limit was applied
    warmup_steps = 0 if limit and len(train_ds) < 1000 else 500

    # Training arguments (use eval_strategy; evaluation_strategy deprecated)
    # Auto-enable mixed precision on GPU for speed
    use_cuda = torch.cuda.is_available()
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    use_bf16 = False
    if use_cuda and callable(is_bf16_supported):
        use_bf16 = bool(is_bf16_supported())
    use_fp16 = bool(use_cuda and not use_bf16)

    settings = get_settings()
    tags = {
        "model": "transformer",
        "model_name": model_name,
        "classifier_version": settings.classifier_version,
        "backend": "huggingface",
    }

    with mlflow_run("transformer", tags=tags) as mlflow_ctx:
        report_targets = ["mlflow"] if mlflow_ctx else ["none"]

        training_args = TrainingArguments(
            output_dir=f"{output_dir}/checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=2,
            dataloader_pin_memory=use_cuda,
            report_to=report_targets,
            bf16=use_bf16,
            fp16=use_fp16,
            optim=("adamw_torch_fused" if use_cuda else "adamw_torch"),
        )

        if mlflow_ctx:
            param_payload = {
                "model_name": model_name,
                "output_dir": output_dir,
                "epochs": epochs,
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "grad_accum": grad_accum,
                "limit": limit if limit is not None else "full",
                "train_samples": len(train_ds),
                "eval_samples": len(test_ds),
                "warmup_steps": warmup_steps,
                "use_bf16": use_bf16,
                "use_fp16": use_fp16,
            }
            mlflow_ctx.log_params(
                {k: str(v) for k, v in param_payload.items()}
            )

        # Initialize trainer with dynamic padding collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=test_tokenized,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Final evaluation
        results = trainer.evaluate()
        print(f"Final results: {results}")

        # Save model and tokenizer
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(f"{output_dir}/best")
        tokenizer.save_pretrained(f"{output_dir}/best")

        # Save label mapping
        label_meta = {
            "label_names": label_names,
            "id2label": {i: name for i, name in enumerate(label_names)},
            "label2id": {name: i for i, name in enumerate(label_names)},
        }

        with open(f"{output_dir}/label_meta.json", "w", encoding="utf-8") as f:
            json.dump(label_meta, f, indent=2)

        # Save metrics
        with open(f"{output_dir}/metrics.txt", "w", encoding="utf-8") as f:
            for key, value in results.items():
                f.write(f"{key}={value:.4f}\n")

        if mlflow_ctx:
            for key, value in results.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    mlflow_ctx.log_metric(key, float(value))
            if trainer.state.best_metric is not None:
                mlflow_ctx.log_metric(
                    "best_metric",
                    float(trainer.state.best_metric),
                )
            if trainer.state.best_model_checkpoint:
                mlflow_ctx.log_param(
                    "best_model_checkpoint",
                    trainer.state.best_model_checkpoint,
                )
            mlflow_ctx.log_metric("train_samples", len(train_ds))
            mlflow_ctx.log_metric("eval_samples", len(test_ds))
            mlflow_ctx.log_artifacts(output_dir)
            mlflow_ctx.log_dict(label_meta, "label_meta.json")

        return results


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="distilbert-base-uncased")
    ap.add_argument("--output-dir", default="models/transformer_agnews")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_transformer_classifier(
        args.model_name,
        args.output_dir,
        args.limit,
        args.epochs,
        args.train_batch_size,
        args.eval_batch_size,
        args.grad_accum,
    )
