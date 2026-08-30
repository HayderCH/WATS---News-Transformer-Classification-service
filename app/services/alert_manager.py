"""
Alert Management Service for Streaming Anomalies
Feature 4: Handles notifications and alerts for detected anomalies
"""

import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional
import json

from app.services.streaming import StreamedArticle

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alerts and notifications for streaming anomalies.
    Supports email, webhook, and logging-based alerts.
    """

    def __init__(self):
        # Alert configuration
        self.email_config = {
            "enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "",
            "password": "",
            "from_email": "",
            "to_emails": [],
        }

        self.webhook_config = {"enabled": False, "url": "", "headers": {}}

        # Alert thresholds and settings
        self.alert_cooldown = 300  # 5 minutes between similar alerts
        self.max_alerts_per_hour = 10

        # Alert history to prevent spam
        self.recent_alerts = []
        self.alert_counts = {}

        logger.info("AlertManager initialized")

    def configure_email(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: list,
    ):
        """Configure email alerts"""
        self.email_config.update(
            {
                "enabled": True,
                "smtp_server": smtp_server,
                "smtp_port": smtp_port,
                "username": username,
                "password": password,
                "from_email": from_email,
                "to_emails": to_emails,
            }
        )
        logger.info("Email alerts configured")

    def configure_webhook(self, url: str, headers: Dict[str, str] = None):
        """Configure webhook alerts"""
        self.webhook_config.update(
            {"enabled": True, "url": url, "headers": headers or {}}
        )
        logger.info("Webhook alerts configured")

    async def send_anomaly_alert(self, article: StreamedArticle):
        """Send alert for detected anomaly"""
        try:
            # Check if we should send this alert (rate limiting)
            if not self._should_send_alert(article):
                return

            alert_data = self._create_alert_data(article)

            # Send alerts via configured channels
            results = []

            if self.email_config["enabled"]:
                result = await self._send_email_alert(alert_data)
                results.append(("email", result))

            if self.webhook_config["enabled"]:
                result = await self._send_webhook_alert(alert_data)
                results.append(("webhook", result))

            # Always log the alert
            self._log_alert(alert_data)

            # Record the alert
            self._record_alert(article)

            logger.info(
                f"Anomaly alert sent: {article.category} "
                f"(channels: {[r[0] for r in results]})"
            )

        except Exception as e:
            logger.error(f"Error sending anomaly alert: {e}")

    def _should_send_alert(self, article: StreamedArticle) -> bool:
        """Check if alert should be sent (rate limiting)"""
        now = datetime.now()

        # Check hourly limit
        hour_key = now.strftime("%Y%m%d%H")
        if hour_key not in self.alert_counts:
            self.alert_counts[hour_key] = 0

        if self.alert_counts[hour_key] >= self.max_alerts_per_hour:
            logger.warning("Alert rate limit exceeded")
            return False

        # Check cooldown for similar alerts
        category = article.category
        for recent_alert in self.recent_alerts[-10:]:  # Check last 10 alerts
            if (
                recent_alert["category"] == category
                and (now - recent_alert["timestamp"]).seconds < self.alert_cooldown
            ):
                logger.info(f"Alert cooldown active for category: {category}")
                return False

        return True

    def _create_alert_data(self, article: StreamedArticle) -> Dict[str, Any]:
        """Create alert data structure"""
        return {
            "alert_type": "anomaly_detected",
            "timestamp": datetime.now().isoformat(),
            "article": {
                "id": article.id,
                "title": article.title,
                "category": article.category,
                "confidence": article.confidence,
                "source": article.source,
            },
            "anomaly": {
                "score": article.anomaly_score,
                "detected_at": article.timestamp.isoformat(),
            },
            "severity": self._calculate_severity(article),
            "message": f"Anomaly detected in {article.category} category "
            f"(score: {article.anomaly_score:.3f})",
        }

    def _calculate_severity(self, article: StreamedArticle) -> str:
        """Calculate alert severity based on anomaly score"""
        score = abs(article.anomaly_score)

        if score > 2.0:
            return "critical"
        elif score > 1.0:
            return "high"
        elif score > 0.5:
            return "medium"
        else:
            return "low"

    async def _send_email_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send email alert"""
        try:
            if not self.email_config["enabled"]:
                return False

            msg = MIMEMultipart()
            msg["From"] = self.email_config["from_email"]
            msg["To"] = ", ".join(self.email_config["to_emails"])
            msg["Subject"] = (
                f"🚨 News Anomaly Alert: {alert_data['article']['category']}"
            )

            body = f"""
News Stream Anomaly Detected

Category: {alert_data['article']['category']}
Title: {alert_data['article']['title']}
Anomaly Score: {alert_data['anomaly']['score']:.3f}
Severity: {alert_data['severity']}
Timestamp: {alert_data['timestamp']}

This alert was automatically generated by the News Classification Streaming Service.
            """

            msg.attach(MIMEText(body, "plain"))

            server = smtplib.SMTP(
                self.email_config["smtp_server"], self.email_config["smtp_port"]
            )
            server.starttls()
            server.login(self.email_config["username"], self.email_config["password"])
            text = msg.as_string()
            server.sendmail(
                self.email_config["from_email"], self.email_config["to_emails"], text
            )
            server.quit()

            return True

        except Exception as e:
            logger.error(f"Email alert failed: {e}")
            return False

    async def _send_webhook_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send webhook alert"""
        try:
            if not self.webhook_config["enabled"]:
                return False

            import aiohttp

            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    **self.webhook_config["headers"],
                }

                async with session.post(
                    self.webhook_config["url"], json=alert_data, headers=headers
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")
            return False

    def _log_alert(self, alert_data: Dict[str, Any]):
        """Log alert to application logs"""
        logger.warning(
            "ANOMALY ALERT: {} - {} (severity: {})".format(
                alert_data["article"]["category"],
                alert_data["message"],
                alert_data["severity"],
            )
        )

    def _record_alert(self, article: StreamedArticle):
        """Record alert for rate limiting"""
        now = datetime.now()

        # Record in recent alerts
        self.recent_alerts.append(
            {
                "category": article.category,
                "timestamp": now,
                "score": article.anomaly_score,
            }
        )

        # Keep only recent alerts (last 24 hours)
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self.recent_alerts = [a for a in self.recent_alerts if a["timestamp"] > cutoff]

        # Update hourly count
        hour_key = now.strftime("%Y%m%d%H")
        self.alert_counts[hour_key] = self.alert_counts.get(hour_key, 0) + 1

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        now = datetime.now()
        hour_key = now.strftime("%Y%m%d%H")

        return {
            "total_alerts_today": len(self.recent_alerts),
            "alerts_this_hour": self.alert_counts.get(hour_key, 0),
            "email_enabled": self.email_config["enabled"],
            "webhook_enabled": self.webhook_config["enabled"],
            "recent_categories": list(
                set(a["category"] for a in self.recent_alerts[-10:])
            ),
        }

    async def test_alert(self, test_type: str = "log") -> bool:
        """Send test alert to verify configuration"""
        test_article = StreamedArticle(
            id="test_alert",
            text="Test anomaly detection alert",
            title="Test Alert",
            category="TEST",
            timestamp=datetime.now(),
            is_anomaly=True,
            anomaly_score=1.5,
        )

        if test_type == "email" and self.email_config["enabled"]:
            alert_data = self._create_alert_data(test_article)
            alert_data["message"] = "TEST ALERT - This is a test notification"
            return await self._send_email_alert(alert_data)
        elif test_type == "webhook" and self.webhook_config["enabled"]:
            alert_data = self._create_alert_data(test_article)
            alert_data["message"] = "TEST ALERT - This is a test notification"
            return await self._send_webhook_alert(alert_data)
        else:
            # Log test
            logger.info("TEST ALERT: News streaming anomaly detection is working")
            return True


# Global alert manager instance
alert_manager = AlertManager()
