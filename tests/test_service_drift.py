"""Test script for BentoML service with drift detection."""

from scripts.serve import NewsClassifierService, ClassificationRequest


def test_service_with_drift():
    """Test the service with drift detection."""
    # Initialize service
    service = NewsClassifierService()

    # Test classification with drift detection
    test_texts = [
        "Apple announces new iPhone with advanced features",
        "Tesla stock rises after earnings report",
        "New COVID variant detected in laboratory",
        "Federal Reserve announces interest rate decision",
        "Climate change summit reaches new agreements",
    ]

    print("Testing BentoML service with drift detection:")
    print("=" * 50)

    for text in test_texts:
        request = ClassificationRequest(text=text, backend="ensemble")
        response = service.classify(request)

        print(f"\nText: {text[:50]}...")
        print(f"Category: {response.top_category}")
        print(f"Confidence: {response.confidence_score:.3f}")
        print(f"Drift Detected: {response.drift_detected}")
        print(f"Drift Score: {response.drift_score:.3f}")
        if response.drift_detected:
            print("Drift Report:")
            print(response.drift_report)


if __name__ == "__main__":
    test_service_with_drift()
