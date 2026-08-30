#!/usr/bin/env python3
"""Test script for the Local NewsChatbot service."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from app.services.chatbot.local_chatbot import chatbot

    print("✅ Local chatbot service imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_local_chatbot_basic():
    """Test basic local chatbot functionality."""
    print("\nTesting Local NewsChatbot basic functionality...")

    # Check if models are loaded
    if not chatbot.generator:
        print("❌ Language model not loaded. Cannot test.")
        return

    # Test intent classification
    query = "What are the latest news about politics?"
    intent, confidence = chatbot.classify_intent(query)
    print(f"Query: '{query}'")
    print(f"Intent: {intent.value}, Confidence: {confidence:.2f}")

    # Test general chat (should work without vector stores)
    print("\nTesting general chat...")
    response = chatbot.chat("Hello, how are you?", session_id="test_session")
    print(f"Response: {response['response'][:200]}...")
    print(f"Intent: {response['intent']}, Confidence: {response['confidence']}")

    print("\n✅ Local chatbot basic test completed!")


if __name__ == "__main__":
    test_local_chatbot_basic()
