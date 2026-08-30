from dataclasses import dataclass
from datetime import datetime


@dataclass
class StreamedArticle:
    id: str
    text: str
    title: str
    category: str
    timestamp: datetime
    source: str = "simulated"
    confidence: float = 0.0
    is_anomaly: bool = False
    anomaly_score: float = 0.0
