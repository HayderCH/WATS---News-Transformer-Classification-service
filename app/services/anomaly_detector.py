"""
Anomaly Detection Service for Streaming News Articles
Feature 4: Detects unusual patterns in news article streams
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.models.streaming import StreamedArticle

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection"""

    is_anomaly: bool
    anomaly_score: float
    confidence: float
    reason: str


class AnomalyDetector:
    """
    Detects anomalies in streaming news articles using statistical and ML methods.
    """

    def __init__(self):
        # Statistical anomaly detection
        self.category_counts = {}
        self.recent_articles = []
        self.window_size = 100  # Rolling window for statistics

        # ML-based anomaly detection
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Anomaly thresholds
        self.statistical_threshold = 3.0  # Standard deviations
        self.ml_threshold = -0.5  # Isolation Forest score threshold

        # Category baseline (expected frequencies)
        self.category_baselines = {
            "POLITICS": 0.15,
            "WELLNESS": 0.08,
            "ENTERTAINMENT": 0.07,
            "TRAVEL": 0.04,
            "STYLE & BEAUTY": 0.04,
            "PARENTING": 0.04,
            "HEALTHY LIVING": 0.04,
            "QUEER VOICES": 0.03,
            "FOOD & DRINK": 0.03,
            "BUSINESS": 0.03,
            "COMEDY": 0.02,
            "SPORTS": 0.02,
            "BLACK VOICES": 0.02,
            "HOME & LIVING": 0.02,
            "PARENTS": 0.02,
            # Default for others
        }

        logger.info("AnomalyDetector initialized")

    def detect(self, article: StreamedArticle) -> Dict[str, Any]:
        """
        Detect anomalies in the article stream.

        Returns:
            dict: Anomaly detection results
        """
        try:
            # Update rolling statistics
            self._update_statistics(article)

            anomaly_results = []

            # Statistical anomaly detection
            stat_result = self._detect_statistical_anomaly(article)
            anomaly_results.append(stat_result)

            # ML-based anomaly detection (if trained)
            if self.is_trained:
                ml_result = self._detect_ml_anomaly(article)
                anomaly_results.append(ml_result)

            # Combine results
            final_result = self._combine_anomaly_results(anomaly_results)

            return {
                "is_anomaly": final_result.is_anomaly,
                "anomaly_score": final_result.anomaly_score,
                "confidence": final_result.confidence,
                "reason": final_result.reason,
            }

        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "confidence": 0.0,
                "reason": "detection_error",
            }

    def _update_statistics(self, article: StreamedArticle):
        """Update rolling statistics with new article"""
        # Update category counts
        category = article.category
        if category not in self.category_counts:
            self.category_counts[category] = 0
        self.category_counts[category] += 1

        # Maintain rolling window
        self.recent_articles.append(
            {
                "category": category,
                "timestamp": article.timestamp,
                "confidence": article.confidence,
            }
        )

        # Keep only recent articles
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.recent_articles = [
            a for a in self.recent_articles if a["timestamp"] > cutoff_time
        ]

        # Limit window size
        if len(self.recent_articles) > self.window_size:
            self.recent_articles = self.recent_articles[-self.window_size :]

    def _detect_statistical_anomaly(self, article: StreamedArticle) -> AnomalyResult:
        """Detect anomalies using statistical methods"""
        try:
            category = article.category
            total_articles = sum(self.category_counts.values())

            if total_articles < 10:  # Not enough data
                return AnomalyResult(False, 0.0, 0.0, "insufficient_data")

            # Calculate current frequency
            current_freq = self.category_counts.get(category, 0) / total_articles

            # Get expected frequency
            expected_freq = self.category_baselines.get(category, 0.01)  # Default 1%

            # Calculate z-score (how many standard deviations from expected)
            if expected_freq > 0:
                # Simplified z-score calculation
                deviation = abs(current_freq - expected_freq)
                # Use expected frequency as rough estimate of standard deviation
                z_score = deviation / max(expected_freq, 0.005)
            else:
                z_score = 0.0

            is_anomaly = z_score > self.statistical_threshold
            confidence = min(z_score / self.statistical_threshold, 1.0)

            reason = f"statistical_zscore_{z_score:.2f}"

            return AnomalyResult(is_anomaly, z_score, confidence, reason)

        except Exception as e:
            logger.error(f"Statistical anomaly detection error: {e}")
            return AnomalyResult(False, 0.0, 0.0, "statistical_error")

    def _detect_ml_anomaly(self, article: StreamedArticle) -> AnomalyResult:
        """Detect anomalies using machine learning (Isolation Forest)"""
        try:
            if not self.is_trained or self.isolation_forest is None:
                return AnomalyResult(False, 0.0, 0.0, "ml_not_trained")

            # Create feature vector
            features = self._extract_features(article)
            features_scaled = self.scaler.transform([features])

            # Predict anomaly score
            score = self.isolation_forest.decision_function(features_scaled)[0]

            # Isolation Forest: negative scores = anomalies
            is_anomaly = score < self.ml_threshold
            confidence = abs(score)  # Higher absolute value = more confident

            reason = f"ml_isolationforest_{score:.3f}"

            return AnomalyResult(is_anomaly, score, confidence, reason)

        except Exception as e:
            logger.error(f"ML anomaly detection error: {e}")
            return AnomalyResult(False, 0.0, 0.0, "ml_error")

    def _extract_features(self, article: StreamedArticle) -> List[float]:
        """Extract features for ML anomaly detection"""
        # Simple features for now
        features = [
            article.confidence,  # Classification confidence
            len(article.text) / 1000.0,  # Normalized text length
            len(article.title) / 100.0,  # Normalized title length
        ]

        # Add category one-hot encoding (simplified)
        category_features = [0.0] * len(self.category_baselines)
        if article.category in self.category_baselines:
            idx = list(self.category_baselines.keys()).index(article.category)
            category_features[idx] = 1.0

        features.extend(category_features)

        return features

    def _combine_anomaly_results(self, results: List[AnomalyResult]) -> AnomalyResult:
        """Combine multiple anomaly detection results"""
        if not results:
            return AnomalyResult(False, 0.0, 0.0, "no_results")

        # Simple voting: if any detector flags anomaly, it's anomalous
        any_anomaly = any(r.is_anomaly for r in results)

        # Average score and confidence
        avg_score = np.mean([r.anomaly_score for r in results])
        avg_confidence = np.mean([r.confidence for r in results])

        # Combine reasons
        reasons = [r.reason for r in results]
        combined_reason = "|".join(reasons)

        return AnomalyResult(any_anomaly, avg_score, avg_confidence, combined_reason)

    def train_ml_model(self, training_articles: List[StreamedArticle] = None):
        """Train the ML anomaly detection model"""
        try:
            logger.info("Training ML anomaly detection model...")

            # Use provided articles or recent articles
            articles = training_articles or []
            if not articles and self.recent_articles:
                # Convert recent articles to StreamedArticle format
                articles = [
                    StreamedArticle(
                        id=f"train_{i}",
                        text="",
                        title="",
                        category=a["category"],
                        timestamp=a["timestamp"],
                        confidence=a.get("confidence", 0.0),
                    )
                    for i, a in enumerate(self.recent_articles)
                ]

            if len(articles) < 10:
                logger.warning("Not enough training data for ML model")
                return False

            # Extract features
            feature_matrix = []
            for article in articles:
                features = self._extract_features(article)
                feature_matrix.append(features)

            # Scale features
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)

            # Train Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=0.1,  # Expected 10% anomalies
                random_state=42,
                n_estimators=100,
            )

            self.isolation_forest.fit(feature_matrix_scaled)
            self.is_trained = True

            logger.info(f"ML anomaly detection trained on {len(articles)} articles")
            return True

        except Exception as e:
            logger.error(f"ML training error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get anomaly detection statistics"""
        return {
            "total_articles_processed": len(self.recent_articles),
            "categories_tracked": len(self.category_counts),
            "ml_model_trained": self.is_trained,
            "category_counts": self.category_counts.copy(),
        }


# Global anomaly detector instance
anomaly_detector = AnomalyDetector()
