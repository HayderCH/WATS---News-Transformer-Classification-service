"""BentoML service for news topic classification."""

import bentoml
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import os
import time
import joblib
from typing import Dict, Any, List
import pandas as pd
from scripts.drift_detection import get_drift_detector


class ClassificationRequest(BaseModel):
    text: str
    backend: str = "ensemble"  # ensemble, sklearn, or transformer


class ClassificationResponse(BaseModel):
    top_category: str
    categories: List[Dict[str, Any]]
    confidence_level: str
    confidence_score: float
    confidence_margin: float
    model_version: str
    latency_ms: float
    suggestion: str
    drift_detected: bool = False
    drift_score: float = 0.0
    drift_report: str = ""


@bentoml.service(
    name="news-topic-classifier",
    resources={"gpu": 0},  # Use CPU for now
    traffic={"timeout": 30},
)
class NewsClassifierService:
    def __init__(self):
        # Load settings from environment
        self.model_dir = Path(os.getenv("MODEL_DIR", "models"))
        self.transformer_model_dir = Path(
            os.getenv("TRANSFORMER_MODEL_DIR", "models/transformer_huffpost")
        )
        self.backend = os.getenv("CLASSIFIER_BACKEND", "ensemble")

        # Label names for HuffPost dataset (42 classes)
        self.label_names = {
            0: "ARTS",
            1: "ARTS_&_CULTURE",
            2: "BLACK_VOICES",
            3: "BUSINESS",
            4: "COLLEGE",
            5: "COMEDY",
            6: "CRIME",
            7: "CULTURE_&_ARTS",
            8: "DIVORCE",
            9: "EDUCATION",
            10: "ENTERTAINMENT",
            11: "ENVIRONMENT",
            12: "FIFTY",
            13: "FOOD_&_DRINK",
            14: "GOOD_NEWS",
            15: "GREEN",
            16: "HEALTHY_LIVING",
            17: "HOME_&_LIVING",
            18: "IMPACT",
            19: "LATINO_VOICES",
            20: "MEDIA",
            21: "MONEY",
            22: "PARENTING",
            23: "PARENTS",
            24: "POLITICS",
            25: "QUEER_VOICES",
            26: "RELIGION",
            27: "SCIENCE",
            28: "SPORTS",
            29: "STYLE",
            30: "STYLE_&_BEAUTY",
            31: "TASTE",
            32: "TECH",
            33: "THE_WORLDPOST",
            34: "TRAVEL",
            35: "U.S._NEWS",
            36: "WEDDINGS",
            37: "WEIRD_NEWS",
            38: "WELLNESS",
            39: "WOMEN",
            40: "WORLD_NEWS",
            41: "WORLDPOST",
        }

        # Initialize drift detector
        self.drift_detector = get_drift_detector()

        # Initialize models
        self._load_models()

    def _load_models(self):
        """Load sklearn and transformer models."""
        # Load sklearn model
        sklearn_dir = self.model_dir / "classifier"
        if sklearn_dir.exists():
            self.vectorizer = joblib.load(sklearn_dir / "tfidf_vectorizer.pkl")
            self.sklearn_model = joblib.load(sklearn_dir / "logreg.pkl")
            meta = joblib.load(sklearn_dir / "label_meta.pkl")
            self.sklearn_classes = meta["classes_"]
        else:
            self.vectorizer = None
            self.sklearn_model = None
            self.sklearn_classes = None

        # Load transformer model
        transformer_dir = self.transformer_model_dir / "best"
        if transformer_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_dir)
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
                transformer_dir
            )
            self.transformer_model.eval()
            # Use CPU for BentoML deployment
            self.device = torch.device("cpu")
            self.transformer_model.to(self.device)
        else:
            self.tokenizer = None
            self.transformer_model = None

    @bentoml.api
    def classify(self, request: ClassificationRequest) -> ClassificationResponse:
        """Classify news text into topics."""
        t_start = time.time()

        # Basic text cleaning
        text = request.text.strip()
        if not text:
            return ClassificationResponse(
                top_category="UNKNOWN",
                categories=[],
                confidence_level="LOW",
                confidence_score=0.0,
                confidence_margin=0.0,
                model_version="unknown",
                latency_ms=(time.time() - t_start) * 1000,
                suggestion="Empty text provided",
            )

        backend = request.backend or self.backend

        if backend == "sklearn" and self.sklearn_model is not None:
            # Sklearn prediction
            vec = self.vectorizer.transform([text])
            probs = self.sklearn_model.predict_proba(vec)[0]

            categories = [
                {
                    "name": self.label_names.get(
                        self.sklearn_classes[i], f"UNKNOWN_{self.sklearn_classes[i]}"
                    ),
                    "prob": float(probs[i]),
                }
                for i in range(len(self.sklearn_classes))
            ]

        elif backend == "transformer" and self.transformer_model is not None:
            # Transformer prediction
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.transformer_model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            categories = [
                {
                    "name": self.label_names.get(i, str(i)),
                    "prob": float(probs[i]),
                }
                for i in range(len(probs))
            ]

        elif (
            backend == "ensemble"
            and self.sklearn_model is not None
            and self.transformer_model is not None
        ):
            # Ensemble: average probabilities
            # Get sklearn probs
            vec = self.vectorizer.transform([text])
            sklearn_probs = self.sklearn_model.predict_proba(vec)[0]

            # Get transformer probs
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.transformer_model(**enc).logits
                tr_probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            # Average probs
            probs = (sklearn_probs + tr_probs) / 2
            categories = [
                {
                    "name": self.label_names.get(i, str(i)),
                    "prob": float(probs[i]),
                }
                for i in range(len(probs))
            ]
        else:
            return ClassificationResponse(
                top_category="ERROR",
                categories=[],
                confidence_level="LOW",
                confidence_score=0.0,
                confidence_margin=0.0,
                model_version="error",
                latency_ms=(time.time() - t_start) * 1000,
                suggestion=f"Backend '{backend}' not available",
            )

        # Sort categories by probability
        categories_sorted = sorted(categories, key=lambda x: x["prob"], reverse=True)
        top_prob = categories_sorted[0]["prob"]

        # Calculate confidence level and margin
        if len(categories_sorted) > 1:
            margin = top_prob - categories_sorted[1]["prob"]
        else:
            margin = 0.0

        if top_prob >= 0.7:
            level = "HIGH"
        elif top_prob >= 0.5:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Generate suggestion
        if level == "LOW":
            suggestion = "Article may span multiple topics"
        elif margin < 0.1:
            suggestion = "Close competition between topics"
        else:
            suggestion = "Clear topic classification"

        latency_ms = (time.time() - t_start) * 1000

        # Check for data drift
        drift_data = pd.DataFrame(
            {"text": [text], "category": [categories_sorted[0]["name"]]}
        )
        drift_results = self.drift_detector.detect_drift(drift_data)
        drift_detected = drift_results.get("dataset_drift", False)
        drift_score = drift_results.get("drift_score", 0.0)
        drift_report = self.drift_detector.get_drift_report(drift_data)

        return ClassificationResponse(
            top_category=categories_sorted[0]["name"],
            categories=categories_sorted[:5],  # Top 5 categories
            confidence_level=level,
            confidence_score=top_prob,
            confidence_margin=margin,
            model_version=f"huffpost-{backend}-v1",
            latency_ms=latency_ms,
            suggestion=suggestion,
            drift_detected=drift_detected,
            drift_score=drift_score,
            drift_report=drift_report,
        )
