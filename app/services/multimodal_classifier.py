"""
Multimodal Classifier Service for News Topic Classification

Combines text and image features for improved classification accuracy.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import requests
import base64
import io

from app.services.classifier import classify_text
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)


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


class MultimodalClassifier:
    """Service for multimodal news classification combining text and images."""

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self.device = self._setup_device(device)
        self.model_path = model_path or "models/fusion/best_model.pth"
        self.fusion_model = None
        self.clip_model = None
        self.clip_processor = None
        self.caption_model = None  # BLIP for image understanding

        # Text embedding dimension (placeholder - should match actual classifier)
        self.text_dim = 768  # BERT/DistilBERT dimension
        self.image_dim = 512  # CLIP ViT-B/32 dimension

        # Initialize CLIP for image embeddings
        self._load_clip_model()

        # Load fusion model if available
        self._load_fusion_model()

        # Load BLIP for image captioning (much better than URL matching!)
        self._load_caption_model()

    def _setup_device(self, device: str) -> str:
        """Setup the compute device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_clip_model(self):
        """Load CLIP model for image embeddings."""
        try:
            logger.info("Loading CLIP model for image embeddings...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.clip_model.to(self.device)
            self.clip_model.eval()
            logger.info(f"CLIP model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.clip_model = None

    def _load_fusion_model(self):
        """Load the trained fusion model."""
        try:
            model_path = Path(self.model_path)
            if model_path.exists():
                logger.info(f"Loading fusion model from {model_path}")

                # Load metrics to get model configuration
                metrics_path = model_path.parent / "metrics.json"
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)

                num_classes = metrics.get(
                    "num_classes", 42
                )  # Default to HuffPost categories

                self.fusion_model = FusionClassifier(
                    text_dim=self.text_dim,
                    image_dim=self.image_dim,
                    num_classes=num_classes,
                )

                state_dict = torch.load(model_path, map_location=self.device)
                self.fusion_model.load_state_dict(state_dict)
                self.fusion_model.to(self.device)
                self.fusion_model.eval()

                logger.info("Fusion model loaded successfully")
            else:
                logger.warning(
                    f"Fusion model not found at {model_path}, using text-only mode"
                )
        except Exception as e:
            logger.error(f"Failed to load fusion model: {e}")
            self.fusion_model = None

    def _load_caption_model(self):
        """Load BLIP model for image captioning and understanding."""
        try:
            logger.info("Loading BLIP model for image captioning...")
            from transformers import pipeline

            self.caption_model = pipeline(
                "image-to-text",
                model="Salesforce/blip-image-captioning-base",
                device=0 if torch.cuda.is_available() else -1,
            )
            logger.info("BLIP captioning model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load BLIP captioning model: {e}")
            self.caption_model = None

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding from the actual classifier model."""
        try:
            # Import here to avoid circular imports
            from app.services.classifier import _classifier_holder

            # Load the classifier if not already loaded
            _classifier_holder.load()

            if (
                _classifier_holder.tr_model is not None
                and _classifier_holder.tr_tokenizer is not None
            ):
                # Use the transformer model to get embeddings
                inputs = _classifier_holder.tr_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                inputs = {k: v.to(_classifier_holder.device) for k, v in inputs.items()}

                with torch.no_grad():
                    # Get the base model outputs (without classification head)
                    outputs = _classifier_holder.tr_model.base_model(**inputs)
                    # Use the [CLS] token embedding as the text representation
                    embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                    return embeddings.cpu().numpy().squeeze().astype(np.float32)
            else:
                # Fallback to sklearn-based approach or random
                logger.warning(
                    "Transformer model not available, using fallback embeddings"
                )
                # For now, use classification probabilities as embedding proxy
                result = classify_text(text)
                if result and "categories" in result:
                    # Create embedding from category probabilities
                    probs = np.array([cat["prob"] for cat in result["categories"]])
                    # Pad or truncate to expected dimension
                    if len(probs) < self.text_dim:
                        probs = np.pad(probs, (0, self.text_dim - len(probs)))
                    else:
                        probs = probs[: self.text_dim]
                    return probs.astype(np.float32)
                else:
                    # Final fallback to deterministic random
                    np.random.seed(hash(text) % 2**32)
                    return np.random.randn(self.text_dim).astype(np.float32)

        except Exception as e:
            logger.warning(f"Failed to get text embedding: {e}, using random fallback")
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(self.text_dim).astype(np.float32)

    def _get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Extract CLIP embedding from image."""
        if self.clip_model is None:
            raise RuntimeError("CLIP model not loaded")

        try:
            with torch.no_grad():
                inputs = self.clip_processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                image_features = self.clip_model.get_image_features(**inputs)

                # Normalize the features
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                return image_features.cpu().numpy().squeeze().astype(np.float32)

        except Exception as e:
            logger.error(f"Failed to extract image embedding: {e}")
            raise

    def _load_image_from_url(self, url: str) -> Image.Image:
        """Load image from URL."""
        try:
            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if not content_type.startswith("image/"):
                raise ValueError(f"URL does not point to an image: {content_type}")

            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return image

        except Exception as e:
            logger.error(f"Failed to load image from URL {url}: {e}")
            raise

    def _load_image_from_base64(self, base64_str: str) -> Image.Image:
        """Load image from base64 string."""
        try:
            # Remove data URL prefix if present
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]

            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return image

        except Exception as e:
            logger.error(f"Failed to load image from base64: {e}")
            raise

    def classify_multimodal(
        self, text: str, image: Optional[Image.Image] = None
    ) -> Dict:
        """
        Classify news article using both text and image (if available).

        Args:
            text: Article text
            image: PIL Image object (optional)

        Returns:
            Classification result with multimodal information
        """
        start_time = time.time()

        # Get text-only classification first
        text_result = classify_text(text)
        modalities = ["text"]

        if image is not None and self.clip_model is not None:
            logger.info(f"Starting multimodal processing with image: {type(image)}")
            try:
                # Extract embeddings
                text_emb = self._get_text_embedding(text)
                image_emb = self._get_image_embedding(image)

                # Convert to tensors
                text_tensor = torch.FloatTensor(text_emb).unsqueeze(0).to(self.device)
                image_tensor = torch.FloatTensor(image_emb).unsqueeze(0).to(self.device)

                # Get fusion prediction
                with torch.no_grad():
                    fusion_logits = self.fusion_model(text_tensor, image_tensor)
                    fusion_probs = torch.softmax(fusion_logits, dim=-1)

                # For now, we'll use the text classifier's categories
                # In production, the fusion model should be trained with proper category mapping
                categories = text_result.get("categories", [])
                if categories:
                    # Update probabilities with fusion results (simplified approach)
                    fusion_probs_np = fusion_probs.cpu().numpy().squeeze()

                    # Blend text and fusion probabilities (weighted average)
                    text_probs = np.array([cat["prob"] for cat in categories])
                    blended_probs = (
                        0.7 * text_probs + 0.3 * fusion_probs_np[: len(text_probs)]
                    )

                    # Debug: log fusion vs text probabilities
                    logger.info(f"Text probs top 5: {text_probs[:5]}")
                    logger.info(f"Fusion probs top 5: {fusion_probs_np[:5]}")
                    logger.info(f"Blended probs top 5: {blended_probs[:5]}")

                    # Normalize
                    blended_probs = blended_probs / blended_probs.sum()

                    # Update categories with blended probabilities
                    for i, cat in enumerate(categories):
                        cat["prob"] = float(blended_probs[i])

                    # Find new top category
                    max_idx = np.argmax(blended_probs)
                    text_result["top_category"] = categories[max_idx]["name"]

                # Set multimodal flags
                text_result["fusion_used"] = True
                text_result["modalities"] = ["text", "image"]

                # Advanced image relevance detection using BLIP captioning
                # Much more sophisticated than simple URL matching!
                image_confidence = self._analyze_image_relevance(image, text)

            except Exception as e:
                logger.warning(
                    "Multimodal classification failed, "
                    f"falling back to text-only: {e}"
                )
                text_result["fusion_used"] = False
                text_result["modalities"] = ["text"]
        else:
            text_result["fusion_used"] = False
            text_result["modalities"] = modalities

        # Add latency
        text_result["latency_ms"] = (time.time() - start_time) * 1000

        # Add image confidence if multimodal processing was used
        if image is not None and self.clip_model is not None:
            text_result["image_confidence"] = image_confidence
        else:
            text_result["image_confidence"] = None

        return text_result

    def _analyze_image_relevance(self, image: Image.Image, text: str) -> float:
        """
        Analyze image relevance to news content using BLIP captioning.
        Much more sophisticated than URL keyword matching!
        """
        try:
            # Use BLIP to generate image caption
            if self.caption_model is not None:
                captions = self.caption_model(image, max_new_tokens=50)
                caption_text = captions[0]["generated_text"].lower()
                logger.info(f"BLIP caption: {caption_text}")

                # Analyze caption for news-relevant content
                confidence = 0.2  # Base confidence

                # High relevance indicators (charts, graphs, financial data)
                chart_keywords = [
                    "chart",
                    "graph",
                    "diagram",
                    "plot",
                    "bar chart",
                    "line graph",
                    "pie chart",
                    "histogram",
                    "trend",
                ]
                if any(keyword in caption_text for keyword in chart_keywords):
                    confidence = 0.9
                    logger.info("High confidence: Chart/graph detected in caption")

                # Financial/business indicators
                finance_keywords = [
                    "stock",
                    "market",
                    "price",
                    "trading",
                    "financial",
                    "business",
                    "economy",
                    "investment",
                    "shares",
                    "index",
                ]
                if any(keyword in caption_text for keyword in finance_keywords):
                    confidence = max(confidence, 0.8)
                    logger.info("High confidence: Financial content detected")

                # News-relevant visual content
                news_keywords = [
                    "news",
                    "headline",
                    "article",
                    "report",
                    "data",
                    "statistics",
                    "analysis",
                    "table",
                    "figure",
                ]
                if any(keyword in caption_text for keyword in news_keywords):
                    confidence = max(confidence, 0.7)
                    logger.info("Good confidence: News-relevant content detected")

                # Check for text content in image (suggests infographics/articles)
                text_indicators = [
                    "text",
                    "writing",
                    "letter",
                    "word",
                    "sentence",
                    "paragraph",
                    "document",
                    "paper",
                ]
                if any(keyword in caption_text for keyword in text_indicators):
                    confidence = max(confidence, 0.6)
                    logger.info("Moderate confidence: Text content detected")

                # Low relevance indicators (photos, illustrations)
                low_relevance = [
                    "person",
                    "people",
                    "portrait",
                    "landscape",
                    "nature",
                    "animal",
                    "building",
                    "street",
                    "car",
                    "food",
                ]
                if any(keyword in caption_text for keyword in low_relevance):
                    confidence = min(confidence, 0.3)
                    logger.info("Low confidence: Non-news visual content")

                return confidence

            else:
                # Fallback to basic analysis if BLIP not available
                logger.warning("BLIP not available, using basic image analysis")
                confidence = 0.3

                # Basic aspect ratio analysis
                try:
                    width, height = image.size
                    if width > height * 1.2:  # Landscape (common for charts)
                        confidence = 0.6
                except Exception:
                    pass

                return confidence

        except Exception as e:
            logger.warning(
                f"Image relevance analysis failed: {e}, using default confidence"
            )
            return 0.3

    def classify_news(
        self,
        title: Optional[str],
        text: str,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> Dict:
        """
        Main classification method supporting text and optional image input.

        Args:
            title: Article title (optional)
            text: Article text
            image_url: URL to article image (optional)
            image_base64: Base64 encoded image (optional)

        Returns:
            Classification result
        """
        # Combine title and text
        full_text = text if title is None else f"{title}. {text}"

        image = None
        if image_url:
            try:
                image = self._load_image_from_url(image_url)
                self._image_url = image_url  # Store for confidence calculation
            except Exception as e:
                logger.warning(f"Failed to load image from URL: {e}")

        elif image_base64:
            try:
                image = self._load_image_from_base64(image_base64)
            except Exception as e:
                logger.warning(f"Failed to load image from base64: {e}")

        return self.classify_multimodal(full_text, image)


# Global instance
_multimodal_classifier = None


def get_multimodal_classifier() -> MultimodalClassifier:
    """Get or create the multimodal classifier instance."""
    global _multimodal_classifier
    if _multimodal_classifier is None:
        _multimodal_classifier = MultimodalClassifier()
    return _multimodal_classifier


def classify_multimodal_news(
    title: Optional[str] = None,
    text: str = "",
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
) -> Dict:
    """
    Convenience function for multimodal news classification.

    Args:
        title: Article title
        text: Article text
        image_url: URL to article image
        image_base64: Base64 encoded image

    Returns:
        Classification result with multimodal information
    """
    classifier = get_multimodal_classifier()
    return classifier.classify_news(title, text, image_url, image_base64)
