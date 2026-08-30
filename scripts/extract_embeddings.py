#!/usr/bin/env python3
"""
CLIP Embedding Extractor for Multi-modal News Classification

This script extracts CLIP embeddings from images for multimodal analysis.
It processes images from the raw directory and saves embeddings to disk.

Usage:
    python scripts/extract_embeddings.py --input-dir data/images/raw
                                         --output-dir data/images/embeddings
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPEmbeddingExtractor:
    """Extracts CLIP embeddings from images."""

    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto"
    ):
        self.model_name = model_name
        self.device = self._setup_device(device)

        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded on device: {self.device}")

    def _setup_device(self, device: str) -> str:
        """Setup the compute device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_metadata(self, metadata_path: Path) -> List[Dict]:
        """Load image metadata."""
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata for {len(metadata)} images")
        return metadata

    def preprocess_image(self, image_path: str) -> Image.Image:
        """Load and preprocess an image."""
        image = Image.open(image_path).convert("RGB")
        return image

    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract CLIP embedding from a single image."""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            image_features = self.model.get_image_features(**inputs)

            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            return image_features.cpu().numpy().squeeze()

    def process_images(
        self, metadata: List[Dict], input_dir: Path
    ) -> Dict[str, np.ndarray]:
        """Process all images and extract embeddings."""
        embeddings = {}

        logger.info("Extracting embeddings from images...")

        for item in tqdm(metadata, desc="Processing images"):
            image_path = Path(item["image_path"])

            # Check if image exists
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            try:
                # Load and process image
                image = self.preprocess_image(str(image_path))

                # Extract embedding
                embedding = self.extract_embedding(image)

                # Store embedding with article ID as key
                article_id = item["article_id"]
                embeddings[article_id] = embedding

            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
                continue

        logger.info(f"Successfully extracted embeddings for {len(embeddings)} images")
        return embeddings

    def save_embeddings(
        self, embeddings: Dict[str, np.ndarray], output_path: Path
    ) -> None:
        """Save embeddings to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy arrays for efficient storage
        embedding_matrix = np.stack(list(embeddings.values()))
        article_ids = list(embeddings.keys())

        # Save as pickle for easy loading
        data = {
            "embeddings": embedding_matrix,
            "article_ids": article_ids,
            "model_name": self.model_name,
            "embedding_dim": embedding_matrix.shape[1],
        }

        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved embeddings to {output_path}")
        logger.info(f"Embedding shape: {embedding_matrix.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings from images")
    parser.add_argument(
        "--input-dir", required=True, help="Input directory with images"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for embeddings"
    )
    parser.add_argument(
        "--model-name", default="openai/clip-vit-base-patch32", help="CLIP model name"
    )
    parser.add_argument(
        "--device", default="auto", help="Compute device (auto/cuda/cpu)"
    )

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    metadata_path = input_dir.parent / "metadata.json"
    embeddings_path = output_dir / "embeddings.pkl"

    # Create extractor
    extractor = CLIPEmbeddingExtractor(args.model_name, args.device)

    # Load metadata
    metadata = extractor.load_metadata(metadata_path)

    # Process images
    embeddings = extractor.process_images(metadata, input_dir)

    # Save embeddings
    extractor.save_embeddings(embeddings, embeddings_path)

    logger.info("Embedding extraction completed!")


if __name__ == "__main__":
    main()
