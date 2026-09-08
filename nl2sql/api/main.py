"""FastAPI Application Entry Point for the NL2SQL Agent Service.

This module sets up the REST API layer:
  - Initializes the FastAPI app with OpenAPI / Swagger documentation (`/docs`).
  - Configures Cross-Origin Resource Sharing (CORS) for web frontends.
  - Mounts modular routers for `/health` and `/chat`.
  - Configures startup and shutdown event hooks.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from nl2sql.api.routes.chat import router as chat_router
from nl2sql.api.routes.health import router as health_router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown hooks."""
    logger.info("🚀 NL2SQL Agent API starting up...")
    yield
    logger.info("🛑 NL2SQL Agent API shutting down...")


# ------------------------------------------------------------------------------
# FastAPI Application Initialization
# ------------------------------------------------------------------------------
app = FastAPI(
    title="NL2SQL Agent API",
    description="A RAG-powered AI agent that translates natural language into SQL for live database querying.",  # noqa: E501
    version="0.1.0",
    docs_url="/docs",    # Interactive Swagger UI endpoint
    redoc_url="/redoc",  # ReDoc documentation endpoint
    lifespan=lifespan,
)

# ------------------------------------------------------------------------------
# Cross-Origin Resource Sharing (CORS) Middleware
# ------------------------------------------------------------------------------
# Enables web applications (e.g. Next.js, React, or local frontends) running on
# different ports or domains to make AJAX/fetch calls to this backend service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to trusted frontend domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Route Registration
# ------------------------------------------------------------------------------
# /health: Checks service liveness and database connectivity
app.include_router(health_router, tags=["Health"])

# /chat: Core conversational endpoint processing user queries through LangGraph
app.include_router(chat_router, prefix="/chat", tags=["Chat"])


@app.get("/")
async def root() -> dict[str, str]:
    """Root discovery endpoint providing links to API documentation and services."""
    return {
        "message": "Welcome to NL2SQL Agent API",
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat",
    }




# Allows direct execution with `python -m nl2sql.api.main`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "nl2sql.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )

