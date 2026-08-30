"""
Streaming API Routes for Real-Time News Processing
Feature 4: REST endpoints for streaming service management
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio

from app.services.streaming import streaming_service, StreamedArticle
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/streaming", tags=["streaming"])


class StreamConfig(BaseModel):
    """Configuration for streaming service"""

    rate: Optional[float] = None  # articles per second


class ManualArticle(BaseModel):
    """Manual article submission"""

    text: str
    title: Optional[str] = None


class AlertConfig(BaseModel):
    """Alert configuration"""

    email_enabled: Optional[bool] = None
    webhook_url: Optional[str] = None
    test_alert: Optional[bool] = None


@router.post("/start")
async def start_streaming(config: Optional[StreamConfig] = None):
    """Start the streaming service"""
    try:
        if config and config.rate:
            streaming_service.set_stream_rate(config.rate)

        success = await streaming_service.start_streaming()

        if success:
            return {
                "status": "started",
                "message": "Streaming service started successfully",
                "config": {
                    "rate": streaming_service.stream_rate,
                    "batch_size": streaming_service.batch_size,
                },
            }
        else:
            raise HTTPException(
                status_code=500, detail="Failed to start streaming service"
            )

    except Exception as e:
        logger.error(f"Error starting streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_streaming():
    """Stop the streaming service"""
    try:
        await streaming_service.stop_streaming()
        return {
            "status": "stopped",
            "message": "Streaming service stopped successfully",
        }
    except Exception as e:
        logger.error(f"Error stopping streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_streaming_status():
    """Get current streaming service status"""
    try:
        stats = streaming_service.get_stats()
        return {
            "active": streaming_service.streaming_active,
            "rate": streaming_service.stream_rate,
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"Error getting streaming status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/article")
async def submit_article(article: ManualArticle, background_tasks: BackgroundTasks):
    """Submit a manual article for processing"""
    try:
        # Process article in background
        background_tasks.add_task(
            streaming_service.process_manual_article, article.text, article.title
        )

        return {"status": "submitted", "message": "Article submitted for processing"}
    except Exception as e:
        logger.error(f"Error submitting article: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config/rate")
async def set_stream_rate(config: StreamConfig):
    """Set streaming rate"""
    try:
        if config.rate is not None:
            streaming_service.set_stream_rate(config.rate)

        return {"status": "updated", "rate": streaming_service.stream_rate}
    except Exception as e:
        logger.error(f"Error setting stream rate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/configure")
async def configure_alerts(config: AlertConfig):
    """Configure alert settings"""
    try:
        alert_manager = streaming_service.alert_manager

        if config.email_enabled is not None:
            # Note: In production, you'd get actual email credentials
            # For demo, we'll just enable/disable
            alert_manager.email_config["enabled"] = config.email_enabled

        if config.webhook_url:
            alert_manager.configure_webhook(config.webhook_url)

        if config.test_alert:
            # Send test alert
            await alert_manager.test_alert()

        return {
            "status": "configured",
            "email_enabled": alert_manager.email_config["enabled"],
            "webhook_enabled": alert_manager.webhook_config["enabled"],
        }
    except Exception as e:
        logger.error(f"Error configuring alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/stats")
async def get_alert_stats():
    """Get alert statistics"""
    try:
        alert_manager = streaming_service.alert_manager
        stats = alert_manager.get_alert_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting alert stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies/stats")
async def get_anomaly_stats():
    """Get anomaly detection statistics"""
    try:
        anomaly_detector = streaming_service.anomaly_detector
        stats = anomaly_detector.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting anomaly stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/anomalies/train")
async def train_anomaly_model():
    """Train the ML anomaly detection model"""
    try:
        anomaly_detector = streaming_service.anomaly_detector
        success = anomaly_detector.train_ml_model()

        if success:
            return {
                "status": "trained",
                "message": "Anomaly detection model trained successfully",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to train anomaly model")

    except Exception as e:
        logger.error(f"Error training anomaly model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def streaming_health_check():
    """Health check for streaming service"""
    try:
        return {
            "status": "healthy",
            "service": "streaming",
            "active": streaming_service.streaming_active,
            "articles_processed": streaming_service.stats["articles_processed"],
            "anomalies_detected": streaming_service.stats["anomalies_detected"],
        }
    except Exception as e:
        logger.error(f"Streaming health check error: {e}")
        return {"status": "unhealthy", "service": "streaming", "error": str(e)}
