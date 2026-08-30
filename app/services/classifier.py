import time
import joblib
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.cuda.amp import autocast
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .preprocessing import basic_clean
from app.core.config import get_settings

_STUB_LABEL_ORDER: List[str] = [
    "WORLD",
    "SPORTS",
    "BUSINESS",
    "SCIENCE_TECH",
]
_STUB_LABEL_TO_IDX: Dict[str, int] = {
    name: idx for idx, name in enumerate(_STUB_LABEL_ORDER)
}
_STUB_KEYWORDS: Dict[str, tuple[str, ...]] = {
    "WORLD": (
        "world",
        "global",
        "international",
        "government",
        "election",
        "diplomat",
    ),
    "SPORTS": (
        "sport",
        "game",
        "match",
        "score",
        "league",
        "team",
        "olympic",
    ),
    "BUSINESS": (
        "stock",
        "market",
        "business",
        "company",
        "earnings",
        "trade",
        "economy",
    ),
    "SCIENCE_TECH": (
        "tech",
        "science",
        "research",
        "space",
        "nasa",
        "rocket",
        "ai",
        "robot",
    ),
}

_LOW_CONFIDENCE_SUGGESTION = "Article may span multiple topics or need more context"
_CLOSE_MARGIN_SUGGESTION = "Close classification - consider human review"

_settings = get_settings()


class _ClassifierHolder:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.classes = None
        self.backend = _settings.classifier_backend
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.tr_tokenizer = None
        self.tr_model = None
        self.stub_active = False
        # Default label names (AG News fallback);
        # will be replaced for transformer
        self.label_names: Dict[int, str] = {
            0: "WORLD",
            1: "SPORTS",
            2: "BUSINESS",
            3: "SCIENCE_TECH",
        }
        # precision flags
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        self.use_bf16 = bool(
            torch.cuda.is_available()
            and callable(is_bf16_supported)
            and is_bf16_supported()
        )
        self.use_fp16 = bool(torch.cuda.is_available() and not self.use_bf16)
        # faster matmul on modern GPUs
        if torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision("high")
            except (AttributeError, RuntimeError):
                pass

    def load(self):
        if self.backend == "stub":
            self.stub_active = True
            return
        if self.backend == "sklearn":
            if self.model is not None:
                return
            model_dir = Path(_settings.model_dir) / "classifier"
            self.vectorizer = joblib.load(model_dir / "tfidf_vectorizer.pkl")
            self.model = joblib.load(model_dir / "logreg.pkl")
            meta = joblib.load(model_dir / "label_meta.pkl")
            self.classes = meta["classes_"]
        elif self.backend == "transformer":
            if self.tr_model is not None:
                return
            tdir = Path(_settings.transformer_model_dir) / "best"
            if not tdir.exists():
                self.backend = "stub"
                self.stub_active = True
                # ensure deterministic mapping order
                self.label_names = {
                    idx: name for idx, name in enumerate(_STUB_LABEL_ORDER)
                }
                return
            self.tr_tokenizer = AutoTokenizer.from_pretrained(
                str(tdir),
                local_files_only=True,
            )  # nosec B615
            model = AutoModelForSequenceClassification.from_pretrained(
                str(tdir),
                local_files_only=True,
            )  # nosec B615
            self.tr_model = model.to(self.device)
            self.tr_model.eval()
            # Attempt to load label mapping if present
            # Fallback to id labels 0..N-1 if not found
            config = self.tr_model.config
            if hasattr(config, "id2label") and config.id2label:
                self.label_names = {
                    int(key): str(value) for key, value in config.id2label.items()
                }
            else:
                self.label_names = {i: str(i) for i in range(config.num_labels)}
        elif self.backend == "ensemble":
            # Load sklearn
            if self.model is None:
                model_dir = Path(_settings.model_dir) / "classifier"
                self.vectorizer = joblib.load(model_dir / "tfidf_vectorizer.pkl")
                self.model = joblib.load(model_dir / "logreg.pkl")
                meta = joblib.load(model_dir / "label_meta.pkl")
                self.classes = meta["classes_"]
            # Load transformer
            if self.tr_model is None:
                tdir = Path(_settings.transformer_model_dir) / "best"
                if not tdir.exists():
                    raise ValueError("Transformer model not found for ensemble")
                self.tr_tokenizer = AutoTokenizer.from_pretrained(
                    str(tdir),
                    local_files_only=True,
                )  # nosec B615
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(tdir),
                    local_files_only=True,
                )  # nosec B615
                self.tr_model = model.to(self.device)
                self.tr_model.eval()
                config = self.tr_model.config
                if hasattr(config, "id2label") and config.id2label:
                    self.label_names = {
                        int(key): str(value) for key, value in config.id2label.items()
                    }
                else:
                    self.label_names = {i: str(i) for i in range(config.num_labels)}

    def _stub_predict(self, text: str) -> tuple[list[Dict[str, Any]], int]:
        lower = text.lower()
        scores: Dict[str, float] = {label: 0.25 for label in _STUB_LABEL_ORDER}
        for label, keywords in _STUB_KEYWORDS.items():
            for keyword in keywords:
                if keyword in lower:
                    scores[label] += 1.0
        if len(lower.split()) > 30:
            scores["WORLD"] += 0.5
        if "technology" in lower or "software" in lower:
            scores["SCIENCE_TECH"] += 0.5
        total = sum(scores.values()) or len(_STUB_LABEL_ORDER)
        categories = [
            {
                "name": label,
                "prob": float(scores[label] / total),
            }
            for label in _STUB_LABEL_ORDER
        ]
        categories_sorted = sorted(
            categories,
            key=lambda data: data["prob"],
            reverse=True,
        )
        top_label = categories_sorted[0]["name"]
        top_idx = _STUB_LABEL_TO_IDX[top_label]
        return categories_sorted, top_idx

    def predict(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        self.load()
        t0 = time.time()
        clean = basic_clean(text)

        if self.backend == "sklearn":
            vec = self.vectorizer.transform([clean])
            probs = self.model.predict_proba(vec)[0]
            top_idx = probs.argmax()
            categories = [
                {
                    "name": self.label_names.get(
                        self.classes[i], f"UNKNOWN_{self.classes[i]}"
                    ),
                    "prob": float(probs[i]),
                }
                for i in range(len(self.classes))
            ]
        elif self.backend == "transformer":
            # transformer backend
            enc = self.tr_tokenizer(
                clean,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                if self.device.type == "cuda":
                    # Use autocast; fall back if dtype arg unsupported
                    try:
                        with autocast(
                            dtype=(torch.bfloat16 if self.use_bf16 else torch.float16)
                        ):
                            logits = self.tr_model(**enc).logits
                    except TypeError:
                        # Older torch autocast signature
                        with autocast():
                            logits = self.tr_model(**enc).logits
                else:
                    logits = self.tr_model(**enc).logits
                # Softmax in float32 for stability and NumPy compatibility
                probs_t = torch.softmax(logits.float(), dim=-1)[0]
                probs = probs_t.detach().to(torch.float32).cpu().numpy()
            top_idx = int(probs.argmax())
            categories = [
                {
                    "name": self.label_names.get(i, str(i)),
                    "prob": float(probs[i]),
                }
                for i in range(len(probs))
            ]
        elif self.backend == "ensemble":
            # Get sklearn probs
            vec = self.vectorizer.transform([clean])
            sklearn_probs = self.model.predict_proba(vec)[0]
            # Get transformer probs
            enc = self.tr_tokenizer(
                clean,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                if self.device.type == "cuda":
                    try:
                        with autocast(
                            dtype=(torch.bfloat16 if self.use_bf16 else torch.float16)
                        ):
                            logits = self.tr_model(**enc).logits
                    except TypeError:
                        with autocast():
                            logits = self.tr_model(**enc).logits
                else:
                    logits = self.tr_model(**enc).logits
                tr_probs = torch.softmax(logits.float(), dim=-1)[0]
                tr_probs = tr_probs.detach().to(torch.float32).cpu().numpy()
            # Average probs
            probs = (sklearn_probs + tr_probs) / 2
            top_idx = int(probs.argmax())
            categories = [
                {
                    "name": self.label_names.get(i, str(i)),
                    "prob": float(probs[i]),
                }
                for i in range(len(probs))
            ]
        else:
            categories, top_idx = self._stub_predict(clean)

        categories_sorted = sorted(
            categories,
            key=lambda x: x["prob"],
            reverse=True,
        )[:top_k]
        latency_ms = (time.time() - t0) * 1000.0

        # Calculate confidence metrics
        top_prob = categories_sorted[0]["prob"]
        second_prob = (
            categories_sorted[1]["prob"] if len(categories_sorted) > 1 else 0.0
        )
        confidence_margin = top_prob - second_prob

        # Determine confidence level
        if top_prob >= 0.8:
            confidence_level = "HIGH"
        elif top_prob >= 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        # Add suggestion for low confidence
        suggestion = None
        if confidence_level == "LOW":
            suggestion = _LOW_CONFIDENCE_SUGGESTION
        elif confidence_margin < 0.1:
            suggestion = _CLOSE_MARGIN_SUGGESTION

        top_name = (
            self.label_names.get(
                self.classes[top_idx], f"UNKNOWN_{self.classes[top_idx]}"
            )
            if self.backend == "sklearn"
            else self.label_names.get(top_idx, str(top_idx))
        )

        result = {
            "top_category": top_name,
            "categories": categories_sorted,
            "confidence_level": confidence_level,
            "confidence_score": float(top_prob),
            "confidence_margin": float(confidence_margin),
            "classifier_version": (
                _settings.classifier_version
                if self.backend != "stub"
                else f"{_settings.classifier_version}-stub"
            ),
            "latency_ms": latency_ms,
        }

        if suggestion:
            result["suggestion"] = suggestion

        return result


_classifier_holder = _ClassifierHolder()


def classify_text(text: str, backend: str = None) -> Dict[str, Any]:
    if backend and backend != _classifier_holder.backend:
        # Create temporary classifier with specified backend
        temp_classifier = _ClassifierHolder()
        temp_classifier.backend = backend
        return temp_classifier.predict(text)
    return _classifier_holder.predict(text)


def classify_batch(items: list[dict], top_k: int = 5) -> list[Dict[str, Any]]:
    """
    Batch classification.
    Each item is a mapping with optional 'title' and required 'text' fields.
    """
    _classifier_holder.load()
    merged = []
    for it in items:
        txt = it.get("text", "")
        title = it.get("title")
        merged.append(basic_clean((f"{title}. {txt}" if title else txt)))

    results: list[Dict[str, Any]] = []
    t_start = time.time()
    if _classifier_holder.backend == "sklearn":
        vec = _classifier_holder.vectorizer.transform(merged)
        probs_mat = _classifier_holder.model.predict_proba(vec)
        for row in probs_mat:
            top_idx = int(row.argmax())
            categories = [
                {
                    "name": _classifier_holder.label_names.get(
                        _classifier_holder.classes[i],
                        f"UNKNOWN_{_classifier_holder.classes[i]}",
                    ),
                    "prob": float(row[i]),
                }
                for i in range(len(_classifier_holder.classes))
            ]
            categories_sorted = sorted(
                categories, key=lambda x: x["prob"], reverse=True
            )[:top_k]
            top_prob = categories_sorted[0]["prob"]
            second_prob = (
                categories_sorted[1]["prob"] if len(categories_sorted) > 1 else 0.0
            )
            margin = top_prob - second_prob
            if top_prob >= 0.8:
                level = "HIGH"
            elif top_prob >= 0.6:
                level = "MEDIUM"
            else:
                level = "LOW"
            suggestion = None
            if level == "LOW":
                suggestion = _LOW_CONFIDENCE_SUGGESTION
            elif margin < 0.1:
                suggestion = _CLOSE_MARGIN_SUGGESTION
            top_name = _classifier_holder.label_names.get(
                _classifier_holder.classes[top_idx],
                f"UNKNOWN_{_classifier_holder.classes[top_idx]}",
            )
            results.append(
                {
                    "top_category": top_name,
                    "categories": categories_sorted,
                    "confidence_level": level,
                    "confidence_score": float(top_prob),
                    "confidence_margin": float(margin),
                    "classifier_version": _settings.classifier_version,
                    "latency_ms": (time.time() - t_start) * 1000.0,
                    "suggestion": suggestion,
                }
            )
    elif _classifier_holder.backend == "transformer":
        # transformer batch
        enc = _classifier_holder.tr_tokenizer(
            merged,
            truncation=True,
            max_length=256,
            return_tensors="pt",
            padding=True,
        ).to(_classifier_holder.device)
        with torch.inference_mode():
            if _classifier_holder.device.type == "cuda":
                try:
                    with autocast(
                        dtype=(
                            torch.bfloat16
                            if _classifier_holder.use_bf16
                            else torch.float16
                        )
                    ):
                        logits = _classifier_holder.tr_model(**enc).logits
                except TypeError:
                    with autocast():
                        logits = _classifier_holder.tr_model(**enc).logits
            else:
                logits = _classifier_holder.tr_model(**enc).logits
            probs_mat = (
                torch.softmax(logits.float(), dim=-1).to(torch.float32).cpu().numpy()
            )
        for row in probs_mat:
            top_idx = int(row.argmax())
            categories = [
                {
                    "name": _classifier_holder.label_names.get(i, str(i)),
                    "prob": float(row[i]),
                }
                for i in range(len(row))
            ]
            categories_sorted = sorted(
                categories, key=lambda x: x["prob"], reverse=True
            )[:top_k]
            top_prob = categories_sorted[0]["prob"]
            second_prob = (
                categories_sorted[1]["prob"] if len(categories_sorted) > 1 else 0.0
            )
            margin = top_prob - second_prob
            if top_prob >= 0.8:
                level = "HIGH"
            elif top_prob >= 0.6:
                level = "MEDIUM"
            else:
                level = "LOW"
            suggestion = None
            if level == "LOW":
                suggestion = _LOW_CONFIDENCE_SUGGESTION
            elif margin < 0.1:
                suggestion = _CLOSE_MARGIN_SUGGESTION
            top_name = _classifier_holder.label_names.get(
                top_idx,
                str(top_idx),
            )
            results.append(
                {
                    "top_category": top_name,
                    "categories": categories_sorted,
                    "confidence_level": level,
                    "confidence_score": float(top_prob),
                    "confidence_margin": float(margin),
                    "classifier_version": _settings.classifier_version,
                    "latency_ms": (time.time() - t_start) * 1000.0,
                    "suggestion": suggestion,
                }
            )
    else:
        for clean in merged:
            categories, top_idx = _classifier_holder._stub_predict(clean)
            categories_sorted = categories[:top_k]
            top_prob = categories_sorted[0]["prob"]
            second_prob = (
                categories_sorted[1]["prob"] if len(categories_sorted) > 1 else 0.0
            )
            margin = top_prob - second_prob
            if top_prob >= 0.8:
                level = "HIGH"
            elif top_prob >= 0.6:
                level = "MEDIUM"
            else:
                level = "LOW"
            suggestion = None
            if level == "LOW":
                suggestion = _LOW_CONFIDENCE_SUGGESTION
            elif margin < 0.1:
                suggestion = _CLOSE_MARGIN_SUGGESTION
            top_name = _classifier_holder.label_names.get(
                top_idx,
                str(top_idx),
            )
            results.append(
                {
                    "top_category": top_name,
                    "categories": categories_sorted,
                    "confidence_level": level,
                    "confidence_score": float(top_prob),
                    "confidence_margin": float(margin),
                    "classifier_version": f"{_settings.classifier_version}-stub",
                    "latency_ms": (time.time() - t_start) * 1000.0,
                    "suggestion": suggestion,
                }
            )
    return results
