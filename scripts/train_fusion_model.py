#!/usr/bin/env python3
"""
Fusion Model Trainer for Multi-modal News Classification

This script trains a fusion classifier that combines text and image embeddings
for improved news topic classification.

Usage:
    python scripts/train_fusion_model.py --embeddings data/images/embeddings.pkl
                                         --output-dir models/fusion/
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import mlflow
import mlflow.pytorch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionDataset(Dataset):
    """Dataset for text-image fusion training."""

    def __init__(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
        labels: np.ndarray,
    ):
        self.text_embeddings = torch.FloatTensor(text_embeddings)
        self.image_embeddings = torch.FloatTensor(image_embeddings)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text_emb": self.text_embeddings[idx],
            "image_emb": self.image_embeddings[idx],
            "label": self.labels[idx],
        }


class FusionClassifier(nn.Module):
    """Simple fusion classifier that concatenates text and image embeddings."""

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.fusion_dim = text_dim + image_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        # Concatenate embeddings
        fused = torch.cat([text_emb, image_emb], dim=-1)
        return self.classifier(fused)


class FusionTrainer:
    """Trainer for the fusion model."""

    def __init__(self, model: nn.Module, device: str = "auto"):
        self.model = model
        self.device = self._setup_device(device)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def _setup_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Training"):
            text_emb = batch["text_emb"].to(self.device)
            image_emb = batch["image_emb"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(text_emb, image_emb)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Evaluate the model."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                text_emb = batch["text_emb"].to(self.device)
                image_emb = batch["image_emb"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(text_emb, image_emb)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)

        # Calculate metrics
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")

        metrics = {
            "loss": avg_loss,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "predictions": all_preds,
            "labels": all_labels,
        }

        return avg_loss, metrics

    def save_model(self, path: Path) -> None:
        """Save the model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)


def load_embeddings(embeddings_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load CLIP embeddings from disk."""
    with open(embeddings_path, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]
    article_ids = data["article_ids"]

    logger.info(
        f"Loaded embeddings: shape {embeddings.shape}, {len(article_ids)} articles"
    )
    return embeddings, article_ids


def load_text_embeddings() -> Tuple[np.ndarray, Dict[str, int]]:
    """Load pre-computed text embeddings (placeholder for now)."""
    # For MVP, we'll use random embeddings to simulate text embeddings
    # In production, this would load actual text embeddings from the classifier
    np.random.seed(42)

    # Load article IDs from metadata to match
    metadata_path = Path("data/images/metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    article_ids = [item["article_id"] for item in metadata]
    num_articles = len(article_ids)

    # Simulate text embeddings (768-dim like BERT)
    text_dim = 768
    text_embeddings = np.random.randn(num_articles, text_dim).astype(np.float32)

    # Create label mapping from metadata
    categories = list(set(item["category"] for item in metadata))
    label_encoder = LabelEncoder()
    label_encoder.fit(categories)

    labels = label_encoder.transform([item["category"] for item in metadata])

    return text_embeddings, labels, label_encoder.classes_


def main():
    parser = argparse.ArgumentParser(
        description="Train fusion model for multimodal classification"
    )
    parser.add_argument(
        "--embeddings", required=True, help="Path to CLIP embeddings pickle"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for model"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="auto", help="Compute device")

    args = parser.parse_args()

    # Setup MLflow
    mlflow.set_experiment("multimodal_fusion")
    with mlflow.start_run():

        # Load embeddings
        image_embeddings, article_ids = load_embeddings(Path(args.embeddings))
        text_embeddings, labels, class_names = load_text_embeddings()

        logger.info(f"Text embeddings shape: {text_embeddings.shape}")
        logger.info(f"Image embeddings shape: {image_embeddings.shape}")
        logger.info(f"Number of classes: {len(class_names)}")

        # Log parameters
        mlflow.log_param("text_embedding_dim", text_embeddings.shape[1])
        mlflow.log_param("image_embedding_dim", image_embeddings.shape[1])
        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)

        # Split data (use simple split if stratification fails due to small classes)
        try:
            train_idx, val_idx = train_test_split(
                range(len(labels)), test_size=0.2, random_state=42, stratify=labels
            )
        except ValueError:
            # Fallback to simple split if stratification fails
            train_idx, val_idx = train_test_split(
                range(len(labels)), test_size=0.2, random_state=42
            )

        train_text = text_embeddings[train_idx]
        train_image = image_embeddings[train_idx]
        train_labels = labels[train_idx]

        val_text = text_embeddings[val_idx]
        val_image = image_embeddings[val_idx]
        val_labels = labels[val_idx]

        # Create datasets
        train_dataset = FusionDataset(train_text, train_image, train_labels)
        val_dataset = FusionDataset(val_text, val_image, val_labels)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Create model
        model = FusionClassifier(
            text_dim=text_embeddings.shape[1],
            image_dim=image_embeddings.shape[1],
            num_classes=len(class_names),
        )

        trainer = FusionTrainer(model, args.device)

        # Training loop
        best_f1 = 0
        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch + 1}/{args.epochs}")

            # Train
            train_loss = trainer.train_epoch(train_loader)

            # Evaluate
            val_loss, val_metrics = trainer.evaluate(val_loader)

            logger.info(".4f")
            logger.info(".4f")

            # Log to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_f1_macro", val_metrics["f1_macro"], step=epoch)
            mlflow.log_metric("val_f1_weighted", val_metrics["f1_weighted"], step=epoch)

            # Save best model
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                model_path = Path(args.output_dir) / "best_model.pth"
                trainer.save_model(model_path)
                mlflow.pytorch.log_model(model, "model")

        # Save final metrics
        final_metrics = {
            "best_f1_macro": best_f1,
            "num_classes": len(class_names),
            "class_names": class_names.tolist(),
            "text_embedding_dim": text_embeddings.shape[1],
            "image_embedding_dim": image_embeddings.shape[1],
        }

        metrics_path = Path(args.output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)

        logger.info("Training completed!")
        logger.info(".4f")


if __name__ == "__main__":
    main()
