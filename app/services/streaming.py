"""
Real-Time Streaming Service for News Classification & Anomaly Detection
Feature 4: Simulated streaming with anomaly detection and alerts
"""

import asyncio
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Callable, Any
import logging

from app.models.streaming import StreamedArticle
from app.services.anomaly_detector import AnomalyDetector
from app.services.alert_manager import AlertManager
from app.services.multimodal_classifier import classify_multimodal_news

logger = logging.getLogger(__name__)


class StreamingService:
    """
    Simulated real-time streaming service for news articles.
    Streams articles with configurable rates and anomaly detection.
    """

    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()

        # Streaming configuration
        self.streaming_active = False
        self.stream_rate = 1.0  # articles per second
        self.batch_size = 10

        # Data source
        self.dataset_path = Path("data/raw/huffpost/" "News_Category_Dataset_v3.json")
        self.articles_data = []

        # Callbacks for real-time processing
        self.article_callbacks: List[Callable[[StreamedArticle], None]] = []
        self.anomaly_callbacks: List[Callable[[StreamedArticle], None]] = []

        # Statistics
        self.stats = {
            "articles_processed": 0,
            "anomalies_detected": 0,
            "start_time": None,
            "categories_count": {},
            "low_confidence_count": 0,
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "processing_rate": 0.0,
            "last_update": None,
        }

        logger.info("StreamingService initialized")

    def load_dataset(self) -> bool:
        """Load articles from the existing dataset for streaming simulation"""
        try:
            if not self.dataset_path.exists():
                logger.error(f"Dataset not found: {self.dataset_path}")
                return False

            logger.info(f"Loading dataset from {self.dataset_path}")

            # Load JSON data (handle large file efficiently)
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                # Read first few lines to understand structure
                sample = ""
                for i, line in enumerate(f):
                    sample += line
                    if i >= 10:  # Read first 10 lines
                        break

                # Try to parse as JSON Lines or single JSON array
                try:
                    # Try single JSON array first
                    f.seek(0)
                    data = json.load(f)
                    if isinstance(data, list):
                        self.articles_data = data
                    else:
                        self.articles_data = [data]
                except json.JSONDecodeError:
                    # Try JSON Lines format
                    f.seek(0)
                    self.articles_data = []
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                self.articles_data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

            logger.info(
                "Loaded {} articles for streaming".format(len(self.articles_data))
            )
            return len(self.articles_data) > 0

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False

    def add_article_callback(self, callback):
        """Add callback for processed articles"""
        self.article_callbacks.append(callback)

    def add_anomaly_callback(self, callback):
        """Add callback for detected anomalies"""
        self.anomaly_callbacks.append(callback)

    def set_stream_rate(self, rate: float):
        """Set streaming rate (articles per second)"""
        # Clamp between 0.1 and 10
        self.stream_rate = max(0.1, min(10.0, rate))
        logger.info(f"Stream rate set to {self.stream_rate} articles/second")

    async def start_streaming(self) -> bool:
        """Start the simulated streaming service"""
        if self.streaming_active:
            logger.warning("Streaming already active")
            return False

        logger.info(f"Starting streaming - articles loaded: {len(self.articles_data)}")
        if not self.articles_data:
            logger.info("Loading dataset...")
            if not self.load_dataset():
                logger.error("Failed to load dataset")
                return False
            logger.info(f"Dataset loaded: {len(self.articles_data)} articles")

        # Ensure classifier is loaded before starting
        from app.services.classifier import _classifier_holder

        _classifier_holder.load()
        logger.info(
            f"Classifier loaded: backend={_classifier_holder.backend}, model={_classifier_holder.tr_model is not None}"
        )

        self.streaming_active = True
        self.stats["start_time"] = datetime.now()

        logger.info(
            "Starting streaming service with rate: {} articles/sec".format(
                self.stream_rate
            )
        )

        # Start background streaming task
        asyncio.create_task(self._streaming_loop())

        return True

    async def stop_streaming(self):
        """Stop the streaming service"""
        self.streaming_active = False
        logger.info("Streaming service stopped")

    async def _streaming_loop(self):
        """Main streaming loop that simulates real-time article processing"""
        article_index = 0

        while self.streaming_active and article_index < len(self.articles_data):
            try:
                # Get batch of articles
                batch_end = min(
                    article_index + self.batch_size, len(self.articles_data)
                )
                batch = self.articles_data[article_index:batch_end]

                # Process batch
                for article_data in batch:
                    await self._process_article(article_data)
                    article_index += 1

                    # Control streaming rate
                    if self.stream_rate < 10:  # Only delay if rate is reasonable
                        await asyncio.sleep(1.0 / self.stream_rate)

                # Small delay between batches
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error

        logger.info("Streaming loop completed")
        self.streaming_active = False

    async def _process_article(self, article_data: Dict[str, Any]):
        """Process a single article through the streaming pipeline"""
        try:
            # Create streamed article object
            article = StreamedArticle(
                id=f"stream_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
                text=article_data.get("short_description", "")
                + " "
                + article_data.get("headline", ""),
                title=article_data.get("headline", ""),
                category="UNKNOWN",  # Start with UNKNOWN, will be classified
                timestamp=datetime.now(),
                source="simulated_dataset",
            )

            # Classify article
            classification_result = await self._classify_article(article)
            article.category = classification_result.get("category", article.category)
            article.confidence = classification_result.get("confidence", 0.0)

            # Check for anomalies
            anomaly_result = await self._detect_anomaly(article)
            article.is_anomaly = anomaly_result.get("is_anomaly", False)
            article.anomaly_score = anomaly_result.get("score", 0.0)

            # Enqueue for human review if low confidence
            await self._enqueue_for_review_if_needed(article)

            # Update statistics
            self._update_stats(article)

            # Trigger callbacks
            for callback in self.article_callbacks:
                try:
                    callback(article)
                except Exception as e:
                    logger.error(f"Error in article callback: {e}")

            # Handle anomalies
            if article.is_anomaly:
                await self._handle_anomaly(article)

            self.stats["articles_processed"] += 1

        except Exception as e:
            logger.error(f"Error processing article: {e}")

    async def _classify_article(self, article: StreamedArticle) -> Dict[str, Any]:
        """Classify article using the multimodal classifier"""
        try:
            # Use the same classifier as the API
            result = classify_multimodal_news(title=article.title, text=article.text)
            return {
                "category": result.get("top_category", "UNKNOWN"),
                "confidence": result.get("confidence_score", 0.0),
            }
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {"category": "UNKNOWN", "confidence": 0.0}

    async def _detect_anomaly(self, article: StreamedArticle) -> Dict[str, Any]:
        """Detect anomalies in the article stream"""
        try:
            result = self.anomaly_detector.detect(article)
            return {
                "is_anomaly": result.get("is_anomaly", False),
                "score": result.get("anomaly_score", 0.0),
            }
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {"is_anomaly": False, "score": 0.0}

    async def _handle_anomaly(self, article: StreamedArticle):
        """Handle detected anomalies (alerts, logging, etc.)"""
        logger.warning(
            f"AnOMALY DETECTED: {article.category} - Score: {article.anomaly_score:.3f}"
        )

        # Trigger anomaly callbacks
        for callback in self.anomaly_callbacks:
            try:
                callback(article)
            except Exception as e:
                logger.error(f"Error in anomaly callback: {e}")

        # Send alert
        try:
            await self.alert_manager.send_anomaly_alert(article)
        except Exception as e:
            logger.error(f"Error sending anomaly alert: {e}")

        self.stats["anomalies_detected"] += 1

    def _update_stats(self, article: StreamedArticle):
        """Update streaming statistics"""
        category = article.category
        confidence = article.confidence

        # Update category count
        if category not in self.stats["categories_count"]:
            self.stats["categories_count"][category] = 0
        self.stats["categories_count"][category] += 1

        # Update confidence distribution
        if confidence >= 0.8:
            self.stats["confidence_distribution"]["high"] += 1
        elif confidence >= 0.6:
            self.stats["confidence_distribution"]["medium"] += 1
        else:
            self.stats["confidence_distribution"]["low"] += 1
            self.stats["low_confidence_count"] += 1

        # Update processing rate
        self.stats["last_update"] = datetime.now()
        if self.stats["start_time"]:
            runtime = datetime.now() - self.stats["start_time"]
            self.stats["processing_rate"] = self.stats["articles_processed"] / max(
                1, runtime.total_seconds()
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get current streaming statistics"""
        stats = self.stats.copy()
        if stats["start_time"]:
            runtime = datetime.now() - stats["start_time"]
            stats["runtime_seconds"] = runtime.total_seconds()
            stats["articles_per_second"] = stats["articles_processed"] / max(
                1, runtime.total_seconds()
            )
        return stats

    async def _enqueue_for_review_if_needed(self, article: StreamedArticle):
        """Enqueue low-confidence streaming articles for human review"""
        try:
            from app.core.config import get_settings

            settings = get_settings()

            # Check if article needs review based on confidence threshold
            if article.confidence < settings.review_conf_threshold:
                # Import here to avoid circular imports
                from app.db.session import SessionLocal
                from app.db.models import ReviewItem

                db = SessionLocal()
                try:
                    # Get top labels from classifier for better review context
                    classification_result = await self._classify_article(article)
                    top_labels = classification_result.get("categories", [])

                    # Create review item
                    review_item = ReviewItem(
                        text=article.text,
                        predicted_label=article.category,
                        confidence_score=article.confidence,
                        confidence_margin=0.0,  # Could calculate if needed
                        model_version="streaming-v1",  # Could be dynamic
                        top_labels=top_labels,
                        source="streaming",
                        stream_id=article.id,
                        anomaly_score=(
                            article.anomaly_score if article.is_anomaly else None
                        ),
                    )

                    db.add(review_item)
                    db.commit()

                    logger.info(
                        f"Enqueued streaming article {article.id} for review "
                        f"(confidence: {article.confidence:.3f})"
                    )

                except Exception as e:
                    logger.error(f"Failed to enqueue streaming article for review: {e}")
                    db.rollback()
                finally:
                    db.close()

        except Exception as e:
            logger.error(f"Error checking review enqueue for streaming article: {e}")

    async def process_manual_article(
        self, text: str, title: str = ""
    ) -> StreamedArticle:
        """Process a manually submitted article"""
        article = StreamedArticle(
            id=f"manual_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
            text=text,
            title=title or text[:100] + "...",
            category="UNKNOWN",
            timestamp=datetime.now(),
            source="manual",
        )

        # Process through pipeline
        await self._process_article(
            {"headline": title, "short_description": text, "category": "UNKNOWN"}
        )

        return article


# Global streaming service instance
streaming_service = StreamingService()
