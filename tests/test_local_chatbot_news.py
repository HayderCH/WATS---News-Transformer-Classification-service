#!/usr/bin/env python3
"""Test script for the Local NewsChatbot with news data."""

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


def test_local_chatbot_with_news():
    """Test local chatbot with news data."""
    print("\n📰 Testing Local NewsChatbot with news data...")

    # Check if models and vector stores are loaded
    if not chatbot.generator:
        print("❌ Language model not loaded.")
        return

    if not chatbot.news_retriever:
        print("❌ News vector store not loaded.")
        return

    # Test news question
    query = "What are the latest news about politics?"
    intent, confidence = chatbot.classify_intent(query)
    print(f"Query: '{query}'")
    print(f"Intent: {intent.value}, Confidence: {confidence:.2f}")

    print("\n🔍 Testing news question with RAG...")
    response = chatbot.chat(query, session_id="test_session")
    print(f"Response: {response['response'][:300]}...")
    intent_val = response["intent"]
    conf_val = response["confidence"]
    print(f"Intent: {intent_val}, Confidence: {conf_val}")
    print(f"Sources: {len(response.get('sources', []))} articles found")

    # Test another news question
    query2 = "Tell me about comedy news"
    print(f"\n🔍 Testing: '{query2}'")
    response2 = chatbot.chat(query2, session_id="test_session")
    print(f"Response: {response2['response'][:300]}...")
    print(f"Sources: {len(response2.get('sources', []))} articles found")

    print("\n✅ News chatbot test completed!")


if __name__ == "__main__":
    test_local_chatbot_with_news()
