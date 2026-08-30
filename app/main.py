from fastapi import FastAPI
from app.core.metrics import metrics_middleware
from app.core.logging import configure_logging, logging_middleware
from app.api.routes.health import router as health_router
from app.api.routes.classify import router as classify_router
from app.api.routes.summarize import router as summarize_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.classify_batch import router as classify_batch_router
from app.api.routes.labels import router as labels_router
from app.api.routes.feedback import router as feedback_router
from app.api.routes.review import router as review_router
from app.api.routes.trends import router as trends_router
from app.api.routes.ab_test import router as ab_test_router
from app.api.routes.images import router as images_router
from app.api.routes.streaming import router as streaming_router
from app.api.routes.chatbot import router as chatbot_router
from app.core.config import get_settings
from app.services.classifier import _classifier_holder

settings = get_settings()
configure_logging()

# Pre-load classifier to ensure it's ready
_classifier_holder.load()

app = FastAPI(title=settings.app_name)
app.middleware("http")(logging_middleware)
app.middleware("http")(metrics_middleware)

app.include_router(health_router)
app.include_router(classify_router)
app.include_router(summarize_router)
app.include_router(metrics_router)
app.include_router(classify_batch_router)
app.include_router(labels_router)
app.include_router(feedback_router)
app.include_router(review_router)
app.include_router(trends_router)
app.include_router(ab_test_router)
app.include_router(images_router, prefix="/images", tags=["images"])
app.include_router(streaming_router)
app.include_router(chatbot_router)


@app.get("/")
def root():
    return {
        "message": "News Topic Intelligence API",
        "endpoints": [
            "/health",
            "/classify_news",
            "/classify_news_batch",
            "/labels",
            "/summarize",
            "/summarize_batch",
            "/metrics",
            "/feedback",
            "/feedback/stats",
            "/review/enqueue",
            "/review/queue",
            "/review/stream-queue",
            "/review/label",
            "/review/stats",
            "/export/dataset",
            "/trends",
            "/ab_test",
            "/ab_test/results/{experiment_name}",
            "/ab_test/complete/{experiment_name}",
            "/ab_test/active",
            "/images/status",
            "/images/generate-image",
            "/images/generate-news-image",
            "/streaming/start",
            "/streaming/stop",
            "/streaming/status",
            "/streaming/article",
            "/streaming/health",
            "/chatbot/chat",
            "/chatbot/history/{session_id}",
            "/chatbot/feedback",
            "/chatbot/stats",
            "/chatbot/health",
        ],
    }
