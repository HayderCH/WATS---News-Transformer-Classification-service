#!/usr/bin/env python3
"""Streamlit chatbot interface for the Local NewsChatbot."""

import streamlit as st
import sys
import os

# Add the project root to the path
# Go up from dashboard to project root
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

try:
    from app.services.chatbot.local_chatbot import chatbot

    CHATBOT_AVAILABLE = True
    CHATBOT_ERROR = None
except ImportError as e:
    CHATBOT_AVAILABLE = False
    CHATBOT_ERROR = str(e)


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="News Chatbot", page_icon="📰", layout="wide")

    st.title("📰 News Topic Classification Chatbot")
    st.markdown("Ask me about news articles, trends, or get help!")

    if not CHATBOT_AVAILABLE:
        error_msg = f"Chatbot service is not available. Error: {CHATBOT_ERROR}"
        st.error(error_msg)
        return

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_content = (
            "Hello! I'm your news chatbot. I can help you with:\n\n"
            "• Latest news articles and trends\n"
            "• Platform documentation and features\n"
            "• Review queue analytics\n"
            "• General questions about news topics\n\n"
            "What would you like to know?"
        )
        welcome_msg = {"role": "assistant", "content": welcome_content}
        st.session_state.messages.append(welcome_msg)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about news..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chatbot.chat(prompt, session_id="streamlit_session")

                    # Display response
                    st.markdown(response["response"])

                    # Show intent and confidence
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Intent: {response['intent']}")
                    with col2:
                        st.caption(f"Confidence: {response['confidence']:.2f}")

                    # Show sources if available
                    if response.get("sources"):
                        num_sources = len(response["sources"])
                        with st.expander(f"📚 Sources ({num_sources})"):
                            for i, source in enumerate(response["sources"], 1):
                                st.markdown(f"**{i}. {source['title']}**")
                                cat = source.get("category", "N/A")
                                st.caption(f"Category: {cat}")
                                if source.get("short_description"):
                                    desc = source["short_description"]
                                    st.text(desc[:200] + "...")
                                st.markdown("---")

                except Exception as e:
                    st.error(f"Error getting response: {e}")

        # Add assistant response to history
        response_text = (
            response["response"]
            if "response" in locals()
            else "Sorry, I encountered an error."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

    # Sidebar with info
    with st.sidebar:
        st.header("🤖 Chatbot Info")
        st.markdown("**Model:** Local (DialoGPT-medium)")
        st.markdown("**Data:** 1000+ news articles")
        st.markdown("**Features:** RAG, Intent Classification")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
