#!/usr/bin/env python3def test_chatbot_basic():
    """Test basic chatbot functionality."""
    print("Testing NewsChatbot basic functionality...")

    # Check if API key is available
    if not chatbot.llm:
        print("⚠️  OpenAI API key not found. Testing intent classification only.")
        # Test intent classification
        query = "What are the latest news about politics?"
        intent, confidence = chatbot.classify_intent(query)
        print(f"Query: '{query}'")
        print(f"Intent: {intent.value}, Confidence: {confidence:.2f}")

        # Test chat (should handle gracefully without LLM)
        response = chatbot.chat(query, session_id="test_session")
        print(f"Response: {response['response'][:100]}...")
        print("Basic test completed (without LLM)!")
        return

    # Full test with API key
    # Test intent classification
    query = "What are the latest news about politics?"
    intent, confidence = chatbot.classify_intent(query)
    print(f"Query: '{query}'")
    print(f"Intent: {intent.value}, Confidence: {confidence:.2f}")

    # Test chat without vector stores (should handle gracefully)
    response = chatbot.chat(query, session_id="test_session")
    print(f"Response: {response['response'][:100]}...")
    print(f"Intent: {response['intent']}, "
          f"Confidence: {response['confidence']}")

    print("Basic test completed successfully!")or the NewsChatbot service."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from app.services.chatbot import chatbot
    print("✅ Chatbot service imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Installing missing dependencies...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "langchain-openai", "langchain-community"], check=True)
    print("Please run the test again after installation.")
    sys.exit(1)


def test_chatbot_basic():
    """Test basic chatbot functionality."""
    print("\nTesting NewsChatbot basic functionality...")

    # Test intent classification
    query = "What are the latest news about politics?"
    intent, confidence = chatbot.classify_intent(query)
    print(f"Query: '{query}'")
    print(f"Intent: {intent.value}, Confidence: {confidence:.2f}")

    # Test chat without vector stores (should handle gracefully)
    response = chatbot.chat(query, session_id="test_session")
    print(f"Response: {response['response'][:100]}...")
    print(f"Intent: {response['intent']}, "
          f"Confidence: {response['confidence']}")

    print("Basic test completed successfully!")


if __name__ == "__main__":
    test_chatbot_basic()

