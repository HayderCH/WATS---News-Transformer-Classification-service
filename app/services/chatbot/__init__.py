"""Multi-Source RAG Chatbot Service for News Intelligence Platform."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class IntentType(Enum):
    """Types of user intents for routing."""

    NEWS_QUESTION = "news_question"
    PLATFORM_HELP = "platform_help"
    CLASSIFICATION_INSIGHT = "classification_insight"
    ANALYTICS_QUERY = "analytics_query"
    GENERAL_CHAT = "general_chat"


class NewsChatbot:
    """Multi-source RAG chatbot for news intelligence platform."""

    def __init__(self):
        """Initialize the chatbot with all components."""
        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Initialize LLM and embeddings if API key is available
        if openai_api_key:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",  # Cost-effective for development
                temperature=0.1,
                api_key=openai_api_key,
            )

            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small", api_key=openai_api_key
            )
        else:
            self.llm = None
            self.embeddings = None

        # Initialize vector stores (will be loaded/populated later)
        self.news_vectorstore = None
        self.docs_vectorstore = None

        # Initialize retrievers
        self.news_retriever = None
        self.docs_retriever = None

        # Conversation memory (simple in-memory for now)
        self.conversation_memory = {}

    def initialize_vector_stores(self) -> None:
        """Initialize or load vector stores for different data sources."""
        # News articles vector store
        news_db_path = "data/vectorstores/news_articles"
        if os.path.exists(news_db_path):
            self.news_vectorstore = Chroma(
                persist_directory=news_db_path, embedding_function=self.embeddings
            )
            self.news_retriever = self.news_vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )

        # Documentation vector store
        docs_db_path = "data/vectorstores/documentation"
        if os.path.exists(docs_db_path):
            self.docs_vectorstore = Chroma(
                persist_directory=docs_db_path, embedding_function=self.embeddings
            )
            self.docs_retriever = self.docs_vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )

    def classify_intent(self, query: str) -> Tuple[IntentType, float]:
        """Classify user intent from query text.

        Returns:
            Tuple of (intent_type, confidence_score)
        """
        # Simple rule-based classification for now
        # TODO: Replace with ML-based classifier in Phase 2

        query_lower = query.lower()

        # News-related keywords
        news_keywords = [
            "article",
            "news",
            "story",
            "headline",
            "recent",
            "latest",
            "politics",
            "sports",
            "business",
            "entertainment",
            "tech",
            "climate",
            "environment",
            "health",
            "science",
        ]

        # Platform help keywords
        help_keywords = [
            "how",
            "what",
            "use",
            "setup",
            "configure",
            "install",
            "api",
            "endpoint",
            "dashboard",
            "streamlit",
            "fastapi",
            "model",
            "training",
            "predict",
            "classify",
        ]

        # Classification insight keywords
        insight_keywords = [
            "why",
            "because",
            "confidence",
            "prediction",
            "classified",
            "category",
            "label",
            "score",
            "probability",
        ]

        # Analytics keywords
        analytics_keywords = [
            "trend",
            "analytics",
            "statistics",
            "metrics",
            "chart",
            "graph",
            "over time",
            "performance",
            "usage",
        ]

        # Count keyword matches
        news_score = sum(1 for keyword in news_keywords if keyword in query_lower)
        help_score = sum(1 for keyword in help_keywords if keyword in query_lower)
        insight_score = sum(1 for keyword in insight_keywords if keyword in query_lower)
        analytics_score = sum(
            1 for keyword in analytics_keywords if keyword in query_lower
        )

        # Determine intent based on highest score
        scores = {
            IntentType.NEWS_QUESTION: news_score,
            IntentType.PLATFORM_HELP: help_score,
            IntentType.CLASSIFICATION_INSIGHT: insight_score,
            IntentType.ANALYTICS_QUERY: analytics_score,
        }

        max_intent = max(scores, key=scores.get)
        max_score = scores[max_intent]

        # If no clear intent, default to general chat
        if max_score == 0:
            return IntentType.GENERAL_CHAT, 0.5

        # Normalize confidence (simple approach)
        total_score = sum(scores.values())
        confidence = max_score / max(1, total_score)

        return max_intent, confidence

    def chat(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user query and return a response.

        Args:
            query: User's question
            session_id: Optional session identifier for conversation memory

        Returns:
            Dict containing response, intent, confidence, and sources
        """
        try:
            # Classify intent
            intent, confidence = self.classify_intent(query)

            # Route to appropriate handler
            if intent == IntentType.NEWS_QUESTION:
                response_data = self._handle_news_question(query)
            elif intent == IntentType.PLATFORM_HELP:
                response_data = self._handle_platform_help(query)
            elif intent == IntentType.CLASSIFICATION_INSIGHT:
                response_data = self._handle_classification_insight(query)
            elif intent == IntentType.ANALYTICS_QUERY:
                response_data = self._handle_analytics_query(query)
            else:
                response_data = self._handle_general_chat(query)

            # Add metadata
            response_data.update(
                {
                    "intent": intent.value,
                    "confidence": confidence,
                    "session_id": session_id,
                    "query": query,
                }
            )

            # Store in conversation memory if session_id provided
            if session_id:
                self._update_conversation_memory(session_id, query, response_data)

            return response_data

        except Exception as e:
            return {
                "response": (
                    f"I apologize, but I encountered an error: "
                    f"{str(e)}. Please try rephrasing your question."
                ),
                "intent": "error",
                "confidence": 0.0,
                "sources": [],
                "error": True,
            }

    def _handle_news_question(self, query: str) -> Dict[str, Any]:
        """Handle news-related questions using RAG over news articles."""
        if not self.news_retriever:
            return {
                "response": (
                    "I'm sorry, but the news article database is not "
                    "available yet. Please try again later."
                ),
                "sources": [],
            }

        try:
            # Retrieve relevant documents
            docs = self.news_retriever.get_relevant_documents(query)

            if not docs:
                return {
                    "response": "I couldn't find any relevant news articles for your question. Could you try rephrasing it?",
                    "sources": [],
                }

            # Prepare context from retrieved documents
            context_parts = []
            sources = []

            for i, doc in enumerate(docs[:3]):  # Limit to top 3
                content = (
                    doc.page_content[:500] + "..."
                    if len(doc.page_content) > 500
                    else doc.page_content
                )
                context_parts.append(f"Article {i+1}: {content}")

                # Extract metadata for sources
                metadata = doc.metadata
                sources.append(
                    {
                        "title": metadata.get("headline", "Unknown"),
                        "category": metadata.get("category", "Unknown"),
                        "date": metadata.get("date", "Unknown"),
                        "authors": metadata.get("authors", "Unknown"),
                    }
                )

            context = "\n\n".join(context_parts)

            # Create RAG prompt
            prompt = ChatPromptTemplate.from_template(
                """
            You are a helpful news intelligence assistant. Use the following news articles to answer the user's question.
            Provide a comprehensive but concise answer based on the provided context.

            Context from news articles:
            {context}

            User Question: {question}

            Answer:"""
            )

            # Create RAG chain
            rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            response = rag_chain.invoke({"context": context, "question": query})

            return {
                "response": response,
                "sources": sources,
                "data_source": "news_articles",
            }

        except Exception as e:
            return {
                "response": f"I encountered an error while searching news articles: {str(e)}",
                "sources": [],
            }

    def _handle_platform_help(self, query: str) -> Dict[str, Any]:
        """Handle platform help questions using documentation RAG."""
        if not self.docs_retriever:
            return {
                "response": "I'm sorry, but the platform documentation is not available yet. Please check the README.md file for basic information.",
                "sources": [],
            }

        try:
            # Retrieve relevant documentation
            docs = self.docs_retriever.get_relevant_documents(query)

            if not docs:
                return {
                    "response": "I couldn't find relevant documentation for your question. You might want to check the main README.md or individual documentation files in the docs/ folder.",
                    "sources": [],
                }

            # Prepare context from documentation
            context_parts = []
            sources = []

            for i, doc in enumerate(docs[:2]):  # Limit to top 2 docs
                content = (
                    doc.page_content[:800] + "..."
                    if len(doc.page_content) > 800
                    else doc.page_content
                )
                context_parts.append(f"Documentation {i+1}: {content}")

                # Extract metadata for sources
                metadata = doc.metadata
                sources.append(
                    {
                        "file": metadata.get("source", "Unknown"),
                        "section": metadata.get("section", "General"),
                    }
                )

            context = "\n\n".join(context_parts)

            # Create help-focused prompt
            prompt = ChatPromptTemplate.from_template(
                """
            You are a helpful assistant for the News Topic Intelligence platform. Use the following documentation to answer the user's question about using the platform.

            Documentation context:
            {context}

            User Question: {question}

            Provide a clear, step-by-step answer based on the documentation:"""
            )

            # Create RAG chain
            rag_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            response = rag_chain.invoke({"context": context, "question": query})

            return {
                "response": response,
                "sources": sources,
                "data_source": "documentation",
            }

        except Exception as e:
            return {
                "response": f"I encountered an error while searching documentation: {str(e)}",
                "sources": [],
            }

    def _handle_classification_insight(self, query: str) -> Dict[str, Any]:
        """Handle classification insight questions."""
        # TODO: Implement in Phase 2 - for now, provide basic response
        return {
            "response": "Classification insights are coming soon! This will help explain why articles are classified into specific categories and show confidence scores.",
            "sources": [],
            "data_source": "platform_data",
        }

    def _handle_analytics_query(self, query: str) -> Dict[str, Any]:
        """Handle analytics and trend questions."""
        # TODO: Implement in Phase 2 - for now, provide basic response
        return {
            "response": "Analytics and trend analysis features are coming soon! This will provide insights into news patterns, category trends, and platform performance metrics.",
            "sources": [],
            "data_source": "analytics",
        }

    def _handle_general_chat(self, query: str) -> Dict[str, Any]:
        """Handle general conversation."""
        response = self.llm.invoke(
            f"You are a helpful news intelligence assistant. Answer this general question: {query}"
        )
        return {
            "response": (
                response.content if hasattr(response, "content") else str(response)
            ),
            "sources": [],
            "data_source": "general",
        }

    def _update_conversation_memory(
        self, session_id: str, query: str, response: Dict[str, Any]
    ) -> None:
        """Update conversation memory for the session."""
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []

        self.conversation_memory[session_id].append(
            {
                "query": query,
                "response": response["response"],
                "timestamp": "now",  # TODO: Add proper timestamp
                "intent": response.get("intent"),
            }
        )

        # Keep only last 10 exchanges
        if len(self.conversation_memory[session_id]) > 10:
            self.conversation_memory[session_id] = self.conversation_memory[session_id][
                -10:
            ]

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        return self.conversation_memory.get(session_id, [])


# Global chatbot instance
# Global chatbot instance
chatbot = NewsChatbot()
