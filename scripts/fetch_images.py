#!/usr/bin/env python3
"""
Image Fetcher for Multi-modal News Classification

This script fetches images for articles from URLs or generates
placeholder images for development/testing purposes. It creates
a metadata file tracking the mapping between articles and their
associated images.

Usage:
    python scripts/fetch_images.py --output-dir data/images/raw
                                   --dataset-size 1000
"""

import argparse
import json
import logging
import random
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageFetcher:
    """Handles fetching and processing images for the multimodal dataset."""

    def __init__(self, output_dir: str, max_image_size_mb: float = 10.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_image_size_mb = max_image_size_mb
        self.metadata: List[Dict] = []

        # Placeholder image generation settings
        self.placeholder_colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
        ]

    def load_huffpost_dataset(self) -> pd.DataFrame:
        """Load the HuffPost dataset for article data."""
        dataset_path = Path("data/raw/huffpost/News_Category_Dataset_v3.json")

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        # Load JSON lines format
        articles = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    articles.append(json.loads(line.strip()))

        df = pd.DataFrame(articles)
        logger.info(f"Loaded {len(df)} articles from HuffPost dataset")
        return df

    def generate_placeholder_image(
        self, article_id: str, category: str, width: int = 400, height: int = 300
    ) -> str:
        """Generate a placeholder image for an article."""
        # Create image with random background color
        color = random.choice(self.placeholder_colors)
        img = Image.new("RGB", (width, height), color=color)
        draw = ImageDraw.Draw(img)

        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        # Add category text
        text = f"{category}\nArticle {article_id}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (width - text_width) // 2
        y = (height - text_height) // 2

        draw.text((x, y), text, fill="white", font=font)

        # Save image
        filename = f"placeholder_{article_id}.jpg"
        filepath = self.output_dir / filename
        img.save(filepath, "JPEG", quality=85)

        return str(filepath)

    def fetch_image_from_url(self, url: str, article_id: str) -> Optional[str]:
        """Fetch an image from a URL with error handling."""
        try:
            # Set headers to mimic a browser
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " "AppleWebKit/537.36"
                )
            }

            response = requests.get(url, headers=headers, timeout=10, stream=True)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if not content_type.startswith("image/"):
                logger.warning(
                    f"URL {url} does not point to an image "
                    f"(content-type: {content_type})"
                )
                return None

            # Check file size
            content_length = response.headers.get("content-length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > self.max_image_size_mb:
                    logger.warning(
                        "Image too large: %.2fMB (max: %.1fMB)",
                        size_mb,
                        self.max_image_size_mb,
                    )
                    return None

            # Download image
            filename = f"article_{article_id}_{int(time.time())}.jpg"
            filepath = self.output_dir / filename

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Validate image can be opened
            try:
                with Image.open(filepath) as img:
                    img.verify()
            except Exception as e:
                logger.warning("Downloaded file is not a valid image: %s", e)
                filepath.unlink()  # Delete invalid file
                return None

            return str(filepath)

        except requests.exceptions.RequestException as e:
            logger.warning("Failed to fetch image from %s: %s", url, e)
            return None
        except Exception as e:
            logger.error("Unexpected error fetching image from %s: %s", url, e)
            return None

    def process_articles(self, df: pd.DataFrame, dataset_size: int) -> None:
        """Process articles and fetch/generate images."""
        # Sample articles for the dataset
        sample_df = df.sample(min(dataset_size, len(df)), random_state=42)

        logger.info(f"Processing {len(sample_df)} articles...")

        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
            article_id = f"{idx:06d}"
            category = row["category"]
            headline = row["headline"]

            # For MVP, we'll generate placeholder images since the dataset doesn't have image URLs
            # In production, you would extract image URLs from article content
            image_path = self.generate_placeholder_image(article_id, category)

            # Store metadata
            metadata_entry = {
                "article_id": article_id,
                "category": category,
                "headline": headline,
                "image_path": image_path,
                "image_type": "placeholder",
                "source_url": row.get("link", ""),
                "date": row.get("date", ""),
                "authors": row.get("authors", ""),
            }

            self.metadata.append(metadata_entry)

            # Small delay to avoid overwhelming the system
            time.sleep(0.01)

    def save_metadata(self) -> None:
        """Save metadata to JSON file."""
        metadata_path = self.output_dir.parent / "metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Saved metadata for {len(self.metadata)} articles to {metadata_path}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch images for multimodal news classification"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for images"
    )
    parser.add_argument(
        "--dataset-size", type=int, default=1000, help="Number of articles to process"
    )
    parser.add_argument(
        "--max-image-size-mb", type=float, default=10.0, help="Maximum image size in MB"
    )

    args = parser.parse_args()

    # Create fetcher
    fetcher = ImageFetcher(args.output_dir, args.max_image_size_mb)

    # Load dataset
    df = fetcher.load_huffpost_dataset()

    # Process articles
    fetcher.process_articles(df, args.dataset_size)

    # Save metadata
    fetcher.save_metadata()

    logger.info("Image fetching completed!")


if __name__ == "__main__":
    main()
