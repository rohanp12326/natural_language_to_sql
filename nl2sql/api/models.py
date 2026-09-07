"""Pydantic Data Transfer Objects (DTOs) for API Requests and Responses.

Defines the JSON schemas validated by FastAPI for:
  - Health check verification
  - Chat requests from users
  - Chat responses returned by the agent
"""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema returned by the `/health` endpoint."""

    status: str = Field(..., description="'healthy' or 'unhealthy'")
    database_connected: bool = Field(..., description="True if database ping succeeded")
    message: str = Field(..., description="Human-readable diagnostics description")


class ChatRequest(BaseModel):
    """Schema expected by the `/chat` POST endpoint."""

    message: str = Field(..., description="User's query or conversational reply")
    session_id: str | None = Field(
        None,
        description="Persistent conversation ID. If omitted, a unique ID is generated."
    )


class ChatResponse(BaseModel):
    """Schema returned by the `/chat` POST endpoint."""

    message: str = Field(..., description="The assistant's generated text response")
    session_id: str = Field(..., description="Active session ID for follow-up turns")
    metadata: dict[str, Any] | None = Field(
        None,
        description="Execution details: user_intent, sql_query, safety, syntax, and execution status"
    )

