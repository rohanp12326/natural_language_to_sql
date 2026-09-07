"""Chat API Router with State Persistence and Interrupt Resumption.

This is the central execution bridge connecting HTTP client requests to LangGraph:
  1. Lazy Singleton Pattern: Database, Vector Store, PostgresSaver, and Graph are
     initialized once upon the first incoming request and cached in memory.
  2. Multi-turn Session Memory: Each conversation uses a unique `session_id` mapped
     to a LangGraph `thread_id`. The compiled graph persists state into PostgreSQL tables
     via `PostgresSaver`.
  3. Human-in-the-Loop (HITL) Interruption & Resumption:
     - When a SQL query is generated, the graph pauses at `human_feedback` via `interrupt()`.
     - The API returns the proposed SQL and asks "Should I execute this query? (yes/no)".
     - When the client responds with "yes" or "no", the API detects the paused thread
       and resumes execution seamlessly using `Command(resume=request.message)`.
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from loguru import logger
from psycopg import Connection
from psycopg.rows import dict_row

from nl2sql.agents.graph import create_graph
from nl2sql.api.models import ChatRequest, ChatResponse
from nl2sql.database.postgresql import PostgreSQLConnector
from nl2sql.knowledge_base.vector_store import VectorStore

router = APIRouter()

# ------------------------------------------------------------------------------
# Global Singletons (Lazy Initialized)
# ------------------------------------------------------------------------------
# Storing these at module level prevents re-connecting to the database and re-compiling
# the LangGraph workflow on every single HTTP request.
_db_connector = None
_vector_store = None
_graph = None
_memory = None


def _initialize_components() -> None:
    """Initialize database, vector store, state checkpointer, and compile the agent graph."""
    global _db_connector, _vector_store, _graph, _memory

    # 1. Initialize PostgreSQL connector
    if _db_connector is None:
        logger.info("🔄 Initializing database connector...")
        _db_connector = PostgreSQLConnector(config_path="configs/database.yml")

    # 2. Initialize PGVector store
    if _vector_store is None:
        logger.info("🔄 Initializing vector store...")
        _vector_store = VectorStore(_db_connector)

    # 3. Initialize LangGraph PostgreSQL checkpoint saver for session persistence
    if _memory is None:
        logger.info("🔄 Initializing PostgreSQL memory...")
        conn_string = _db_connector.create_postgresql_uri()
        postgres_conn = Connection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        )
        _memory = PostgresSaver(conn=postgres_conn)
        # Create checkpoints and checkpoint_writes tables in PostgreSQL if they don't exist
        _memory.setup()

    # 4. Compile the LangGraph workflow with the Postgres checkpointer attached
    if _graph is None:
        logger.info("🔄 Initializing agent graph...")
        workflow = create_graph(_db_connector, _vector_store)
        _graph = workflow.compile(checkpointer=_memory)


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle incoming user chat messages, execute agent workflow, and manage interruptions.

    Workflow:
      - Validates / generates session ID.
      - Checks if current thread is paused waiting for user approval.
      - If resuming: calls `graph.invoke(Command(resume=...))`.
      - If new turn: creates HumanMessage and invokes graph.
      - If graph interrupted: returns confirmation prompt.
      - If graph finished: returns final assistant message + metadata.

    Args:
        request: Incoming message payload with text and optional session ID.

    Returns:
        ChatResponse: Assistant response message, active session ID, and pipeline metadata.
    """
    try:
        # Ensure singletons are instantiated
        _initialize_components()

        # Generate a unique session ID if client did not supply one
        session_id = (
            request.session_id
            or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"  # noqa: E501
        )

        logger.info(f"🔄 Processing chat request for session: {session_id}")
        logger.debug(f"User message: {request.message}")

        user_message = HumanMessage(content=request.message)

        # In LangGraph, thread_id partitions state history per user/session
        config = {"configurable": {"thread_id": session_id}}

        # ----------------------------------------------------------------------
        # Check for Interrupted State (HITL Resumption)
        # ----------------------------------------------------------------------
        # Inspect checkpoint storage for any pending interrupted node in this thread
        existing_state = _graph.get_state(config)
        if existing_state.next and request.message.lower() in ["yes", "no", "y", "n"]:
            # The previous graph run paused at `interrupt()` and the user is now replying
            logger.debug("🔄 Resuming interrupted graph with user feedback...")
            from langgraph.types import Command

            # Resume execution passing the user's reply string directly to `interrupt()`
            result = _graph.invoke(Command(resume=request.message), config=config)
        else:
            # New turn: seed initial state with the new user message
            initial_state = {"messages": [user_message]}

            logger.debug("🔄 Executing agent graph...")
            result = _graph.invoke(initial_state, config=config)

        # ----------------------------------------------------------------------
        # Handle Graph Output (Interrupted vs Completed)
        # ----------------------------------------------------------------------
        if "__interrupt__" in result:
            # Execution paused at human_feedback node!
            # Extract the confirmation prompt to return to the user
            interrupt_data = result["__interrupt__"][0].value
            interrupt_message = interrupt_data.get("messages")
            if hasattr(interrupt_message, "content"):
                last_message = interrupt_message.content
            elif isinstance(interrupt_message, str):
                last_message = interrupt_message
            else:
                last_message = (
                    "Waiting for your confirmation. Please respond with 'yes' or 'no'."
                )
        else:
            # Workflow completed successfully. Extract the last AI response.
            last_message = None
            if result.get("messages"):
                for msg in reversed(result["messages"]):
                    if hasattr(msg, "type") and msg.type != "human":
                        last_message = msg.content
                        break

            if last_message is None:
                last_message = (
                    "I processed your message but couldn't generate a response."
                )

        logger.info(f"✅ Chat request completed for session: {session_id}")

        # Return structured response with detailed pipeline metadata for frontend display
        return ChatResponse(
            message=last_message,
            session_id=session_id,
            metadata={
                "user_intent": result.get("user_intent"),
                "sql_query": result.get("sql_query"),
                "sql_safety_status": result.get("sql_safety_status"),
                "sql_syntax_status": result.get("sql_syntax_status"),
                "sql_execution_status": result.get("sql_execution_status"),
            },
        )

    except Exception as e:
        logger.error(f"❌ Chat request failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Chat processing error: {e!s}"
        ) from e

