"""Chatbot API routes for the News Topic Intelligence platform."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.chatbot.local_chatbot import LocalNewsChatbot

# Create router
router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# Global chatbot instance (lazy loaded)
_chatbot_instance: Optional[LocalNewsChatbot] = None


def get_chatbot() -> LocalNewsChatbot:
    """Get or create the global chatbot instance."""
    global _chatbot_instance
    if _chatbot_instance is None:
        try:
            _chatbot_instance = LocalNewsChatbot()
        except Exception as e:
            raise HTTPException(
                status_code=503, detail=f"Chatbot service unavailable: {str(e)}"
            )
    return _chatbot_instance


# Pydantic models for request/response
class ChatRequest(BaseModel):
    """Request model for chatbot chat endpoint."""

    message: str = Field(..., description="The user's message to send to the chatbot")
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for conversation continuity. If not provided, a new session will be created.",
    )
    user_id: Optional[str] = Field(
        None, description="Optional user identifier for analytics"
    )


class ChatResponse(BaseModel):
    """Response model for chatbot chat endpoint."""

    response: str = Field(..., description="The chatbot's response message")
    session_id: str = Field(
        ..., description="The session ID used for this conversation"
    )
    intent: str = Field(..., description="The detected intent of the user's query")
    confidence: float = Field(
        ..., description="Confidence score for the intent classification"
    )
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of sources used to generate the response",
    )
    data_source: str = Field(..., description="The primary data source used")
    timestamp: float = Field(..., description="Unix timestamp of the response")
    request_id: str = Field(..., description="Unique request identifier")


class ConversationHistory(BaseModel):
    """Model for conversation history entries."""

    query: str = Field(..., description="The user's query")
    response: str = Field(..., description="The chatbot's response")
    timestamp: float = Field(..., description="Unix timestamp of the exchange")
    intent: Optional[str] = Field(None, description="Detected intent for the query")


class HistoryResponse(BaseModel):
    """Response model for conversation history endpoint."""

    session_id: str = Field(..., description="The session ID")
    history: List[ConversationHistory] = Field(
        default_factory=list, description="List of conversation exchanges"
    )
    total_exchanges: int = Field(
        ..., description="Total number of exchanges in the session"
    )


class FeedbackRequest(BaseModel):
    """Request model for user feedback on chatbot responses."""

    session_id: str = Field(..., description="The session ID the feedback relates to")
    request_id: str = Field(..., description="The request ID being rated")
    rating: int = Field(
        ..., ge=1, le=5, description="Rating from 1-5 (1=very poor, 5=excellent)"
    )
    feedback_text: Optional[str] = Field(
        None, description="Optional detailed feedback text"
    )
    user_id: Optional[str] = Field(None, description="Optional user identifier")


class StatsResponse(BaseModel):
    """Response model for chatbot usage statistics."""

    total_sessions: int = Field(..., description="Total number of chat sessions")
    total_messages: int = Field(..., description="Total number of messages processed")
    average_response_time: float = Field(
        ..., description="Average response time in seconds"
    )
    intent_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Distribution of intents processed"
    )
    data_source_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Distribution of data sources used"
    )
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


# In-memory storage for demo purposes (in production, use a database)
_chatbot_stats = {
    "total_sessions": 0,
    "total_messages": 0,
    "response_times": [],
    "intent_counts": {},
    "data_source_counts": {},
    "start_time": time.time(),
    "sessions": set(),  # Track unique session IDs
}


@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest) -> ChatResponse:
    """Send a message to the chatbot and get a response.

    This endpoint processes user messages through the multi-source RAG chatbot,
    which can answer questions about news articles, platform documentation,
    classification insights, and analytics.
    """
    start_time = time.time()

    try:
        # Get chatbot instance
        chatbot = get_chatbot()

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid4())

        # Track session
        _chatbot_stats["sessions"].add(session_id)
        _chatbot_stats["total_sessions"] = len(_chatbot_stats["sessions"])

        # Send message to chatbot
        result = chatbot.chat(request.message, session_id=session_id)

        # Track statistics
        response_time = time.time() - start_time
        _chatbot_stats["total_messages"] += 1
        _chatbot_stats["response_times"].append(response_time)

        # Update intent and data source counts
        intent = result.get("intent", "unknown")
        data_source = result.get("data_source", "unknown")

        _chatbot_stats["intent_counts"][intent] = (
            _chatbot_stats["intent_counts"].get(intent, 0) + 1
        )
        _chatbot_stats["data_source_counts"][data_source] = (
            _chatbot_stats["data_source_counts"].get(data_source, 0) + 1
        )

        # Create response
        response = ChatResponse(
            response=result.get("response", ""),
            session_id=session_id,
            intent=intent,
            confidence=result.get("confidence", 0.0),
            sources=result.get("sources", []),
            data_source=data_source,
            timestamp=time.time(),
            request_id=str(uuid4()),
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_conversation_history(session_id: str) -> HistoryResponse:
    """Get the conversation history for a specific session.

    Returns all the conversation exchanges (user queries and bot responses)
    for the specified session ID.
    """
    try:
        chatbot = get_chatbot()
        history = chatbot.get_conversation_history(session_id)

        # Convert to response format
        history_entries = []
        for entry in history:
            history_entries.append(
                ConversationHistory(
                    query=entry["query"],
                    response=entry["response"],
                    timestamp=entry.get("timestamp", time.time()),
                    intent=entry.get("intent"),
                )
            )

        return HistoryResponse(
            session_id=session_id,
            history=history_entries,
            total_exchanges=len(history_entries),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving conversation history: {str(e)}"
        )


@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest) -> Dict[str, str]:
    """Submit user feedback on a chatbot response.

    This endpoint allows users to rate chatbot responses and provide
    detailed feedback to help improve the system.
    """
    # In a real implementation, this would be stored in a database
    # For now, we'll just log it and return success

    print(f"Chatbot feedback received:")
    print(f"  Session ID: {feedback.session_id}")
    print(f"  Request ID: {feedback.request_id}")
    print(f"  Rating: {feedback.rating}/5")
    if feedback.feedback_text:
        print(f"  Feedback: {feedback.feedback_text}")
    if feedback.user_id:
        print(f"  User ID: {feedback.user_id}")

    return {"status": "feedback_received", "message": "Thank you for your feedback!"}


@router.get("/stats", response_model=StatsResponse)
async def get_chatbot_stats() -> StatsResponse:
    """Get usage statistics for the chatbot service.

    Returns metrics about chatbot usage, performance, and behavior patterns.
    """
    # Calculate average response time
    response_times = _chatbot_stats["response_times"]
    avg_response_time = (
        sum(response_times) / len(response_times) if response_times else 0.0
    )

    return StatsResponse(
        total_sessions=_chatbot_stats["total_sessions"],
        total_messages=_chatbot_stats["total_messages"],
        average_response_time=avg_response_time,
        intent_distribution=_chatbot_stats["intent_counts"].copy(),
        data_source_distribution=_chatbot_stats["data_source_counts"].copy(),
        uptime_seconds=time.time() - _chatbot_stats["start_time"],
    )


@router.get("/health")
async def chatbot_health_check() -> Dict[str, Any]:
    """Health check endpoint for the chatbot service."""
    try:
        # Try to get chatbot instance to verify it's working
        chatbot = get_chatbot()

        # Quick test to ensure basic functionality
        test_result = chatbot.chat("hello", session_id="health_check")
        is_healthy = bool(test_result.get("response"))

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "chatbot",
            "timestamp": time.time(),
            "version": "1.0.0",
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "chatbot",
            "error": str(e),
            "timestamp": time.time(),
        }
