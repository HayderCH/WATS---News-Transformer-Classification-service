#!/usr/bin/env python3
"""Test script for BentoML service."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.serve import NewsClassifierService


def test_service():
    print("Testing BentoML service...")
    try:
        service = NewsClassifierService()
        print("✓ Service instantiated successfully")

        # Test classification
        from scripts.serve import ClassificationRequest

        request = ClassificationRequest(
            text="Apple announces new iPhone", backend="ensemble"
        )
        result = service.classify(request)
        print("✓ Classification successful")
        print(
            f"Result: {result.top_category} (confidence: {result.confidence_score:.3f})"
        )

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_service()
