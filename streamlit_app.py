"""Interactive Streamlit Frontend for the NL2SQL Agent.

This application provides a modern web interface for the NL2SQL agent:
  - Interactive multi-turn chat with persistent session memory
  - One-click Human-in-the-Loop (HITL) query execution approval/rejection
  - Formatted SQL code blocks and pipeline execution metrics
  - Interactive tabular view of query results with CSV download export
  - Live API and PostgreSQL database health diagnostics in the sidebar
  - Quick-start sample prompts for instant exploration
"""

import uuid
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------
# Page Configuration & Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="NL2SQL Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished interface styling
st.markdown(
    """
    <style>
    .metric-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        border-radius: 0.35rem;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-success { background-color: #d1fae5; color: #065f46; }
    .badge-info { background-color: #dbeafe; color: #1e40af; }
    .badge-warning { background-color: #fef3c7; color: #92400e; }
    .badge-error { background-color: #fee2e2; color: #991b1b; }
    .hitl-container {
        padding: 1rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------------------
def generate_session_id() -> str:
    """Generate a unique timestamped session ID compatible with LangGraph thread_id."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = str(uuid.uuid4())[:8]
    return f"streamlit_{timestamp}_{suffix}"


if "session_id" not in st.session_state:
    st.session_state.session_id = generate_session_id()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

if "prompt_to_send" not in st.session_state:
    st.session_state.prompt_to_send = None


# ------------------------------------------------------------------------------
# API Communication Helpers
# ------------------------------------------------------------------------------
@st.cache_data(ttl=15, show_spinner=False)
def check_health(base_url: str) -> dict[str, Any]:
    """Check the health status of the FastAPI backend and database."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/health")
            if response.status_code == 200:
                return response.json()
            return {
                "status": "unhealthy",
                "database_connected": False,
                "message": f"HTTP Error {response.status_code}",
            }
    except Exception as e:
        return {
            "status": "unreachable",
            "database_connected": False,
            "message": f"Connection failed: {e!s}",
        }


def send_chat_message(
    base_url: str, message: str, session_id: str
) -> dict[str, Any] | None:
    """Send a user message to the NL2SQL /chat/ endpoint."""
    try:
        with httpx.Client(timeout=90.0) as client:
            response = client.post(
                f"{base_url}/chat/",
                json={"message": message, "session_id": session_id},
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"❌ API Error ({response.status_code}): {response.text}")
                return None
    except httpx.ConnectError:
        st.error(
            f"❌ Unable to connect to API server at `{base_url}`. "
            "Please ensure the backend is running via `python -m nl2sql.api.main`."
        )
        return None
    except httpx.TimeoutException:
        st.error(
            "⏳ Request timed out (90s). The LLM or database query took "
            "longer than expected."
        )
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error communicating with backend: {e}")
        return None


def reset_session() -> None:
    """Reset the current conversation and generate a fresh session ID."""
    st.session_state.session_id = generate_session_id()
    st.session_state.messages = []
    st.session_state.prompt_to_send = None


# ------------------------------------------------------------------------------
# Sidebar: System Diagnostics & Settings
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 NL2SQL Control Panel")
    st.markdown("Query PostgreSQL databases using natural language with AI agents.")

    st.divider()

    # Backend configuration
    st.subheader("⚙️ Configuration")
    base_url_input = st.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
        help="FastAPI backend host and port",
    )
    if base_url_input != st.session_state.api_base_url:
        st.session_state.api_base_url = base_url_input

    # Health check card
    st.subheader("🩺 System Status")
    health = check_health(st.session_state.api_base_url)

    col1, col2 = st.columns(2)
    with col1:
        if health["status"] == "healthy":
            st.success("API: Online", icon="🟢")
        else:
            st.error(f"API: {health['status'].title()}", icon="🔴")

    with col2:
        if health.get("database_connected"):
            st.success("DB: Connected", icon="🟢")
        else:
            st.error("DB: Offline", icon="🔴")

    if health["status"] != "healthy" or not health.get("database_connected"):
        st.warning(
            "Backend is offline. Start it in terminal:\n"
            "```bash\n"
            "python -m nl2sql.api.main\n"
            "```"
        )

    if st.button("🔄 Check Health Now", use_container_width=True):
        check_health.clear()
        st.rerun()

    st.divider()

    # Session management
    st.subheader("💬 Session")
    st.caption(f"**ID:** `{st.session_state.session_id}`")
    if st.button("+ New Conversation", use_container_width=True, type="secondary"):
        reset_session()
        st.rerun()

    st.divider()

    # Quick prompt examples
    st.subheader("💡 Example Queries")
    example_prompts = [
        "Show me the first 5 customers",
        "Count total orders by order status",
        "What are the top 5 product categories by sales?",
        "Show orders with payment value greater than $1000",
        "What tables exist in the database?",
    ]

    for example in example_prompts:
        if st.button(example, use_container_width=True, key=f"ex_{example}"):
            st.session_state.prompt_to_send = example
            st.rerun()


# ------------------------------------------------------------------------------
# Main Chat Display
# ------------------------------------------------------------------------------
st.title("💬 Natural Language to SQL Assistant")
st.caption("Powered by LangGraph, PostgreSQL (Olist E-Commerce), and GLM Reasoning.")

# Render conversation history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # Display main message content
        st.markdown(msg["content"])

        metadata = msg.get("metadata")
        if metadata:
            # Display pipeline badges if available
            badges = []
            if metadata.get("user_intent"):
                intent_str = metadata["user_intent"].upper()
                badges.append(
                    f"<span class='metric-badge badge-info'>"
                    f"Intent: {intent_str}</span>"
                )
            if metadata.get("sql_safety_status"):
                status = metadata["sql_safety_status"]
                cls = "badge-success" if status == "safe" else "badge-error"
                badges.append(
                    f"<span class='metric-badge {cls}'>"
                    f"Safety: {status.title()}</span>"
                )
            if metadata.get("sql_syntax_status"):
                status = str(metadata["sql_syntax_status"])
                cls = "badge-success" if status == "valid" else "badge-error"
                badges.append(
                    f"<span class='metric-badge {cls}'>"
                    f"Syntax: {status.title()}</span>"
                )
            if metadata.get("sql_execution_status"):
                status = metadata["sql_execution_status"]
                cls = "badge-success" if status == "success" else "badge-error"
                badges.append(
                    f"<span class='metric-badge {cls}'>"
                    f"Execution: {status.title()}</span>"
                )

            if badges:
                st.markdown("".join(badges), unsafe_allow_html=True)

            # Display generated SQL in an expander
            sql_query = metadata.get("sql_query")
            if sql_query:
                with st.expander("🔍 Inspect Generated SQL", expanded=False):
                    st.code(sql_query, language="sql")
                    if metadata.get("sql_explanation"):
                        st.markdown(f"**Explanation:** {metadata['sql_explanation']}")

            # Display SQL execution tabular results if records are available
            exec_result = metadata.get("sql_execution_result")
            if exec_result and isinstance(exec_result, dict):
                data_records = exec_result.get("data")
                if (
                    data_records
                    and isinstance(data_records, list)
                    and len(data_records) > 0
                ):
                    df = pd.DataFrame(data_records)
                    row_count = exec_result.get("row_count", len(df))

                    st.markdown(f"**Query Results ({row_count} rows):**")
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=min(300, 35 * (len(df) + 1)),
                    )

                    # CSV download button
                    csv_data = df.to_csv(index=False).encode("utf-8")
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name=f"nl2sql_results_{timestamp_str}.csv",
                        mime="text/csv",
                        key=f"dl_{idx}",
                    )
                elif exec_result.get("error"):
                    st.error(f"Execution Error: {exec_result['error']}")


# ------------------------------------------------------------------------------
# Human-in-the-Loop (HITL) Action Controls
# ------------------------------------------------------------------------------
# Check if the latest assistant message is requesting query approval
last_msg = st.session_state.messages[-1] if st.session_state.messages else None
is_waiting_approval = False

if (
    not st.session_state.prompt_to_send
    and last_msg
    and last_msg["role"] == "assistant"
):
    text_content = last_msg["content"].lower()
    if (
        "execute this query" in text_content
        or "waiting for your confirmation" in text_content
        or "answer with 'yes' or 'no'" in text_content
        or "yes/no" in text_content
    ):
        is_waiting_approval = True

if is_waiting_approval:
    st.markdown("---")
    st.info(
        "✋ **Human Confirmation Required:** "
        "Please confirm whether to execute the generated query."
    )
    col_approve, col_reject, _ = st.columns([1, 1, 3])

    with col_approve:
        if st.button(
            "✅ Approve & Execute (Yes)", type="primary", use_container_width=True
        ):
            st.session_state.prompt_to_send = "yes"
            st.rerun()

    with col_reject:
        if st.button(
            "❌ Reject Query (No)", type="secondary", use_container_width=True
        ):
            st.session_state.prompt_to_send = "no"
            st.rerun()


# ------------------------------------------------------------------------------
# User Input Processing
# ------------------------------------------------------------------------------
chat_input = st.chat_input(
    "Ask a question about your database (e.g. 'Show top 5 customers')..."
)

# Process either typed input or quick prompt button selection
prompt = chat_input or st.session_state.prompt_to_send

if prompt:
    # Clear quick prompt state
    st.session_state.prompt_to_send = None

    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Render user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend API
    is_confirm = prompt.strip().lower() in ["yes", "y", "no", "n"]
    spinner_text = (
        "⚙️ Executing query & analyzing results with AI..."
        if is_confirm
        else "🤖 Thinking & generating SQL..."
    )
    with st.chat_message("assistant"):
        with st.spinner(spinner_text):
            response_data = send_chat_message(
                base_url=st.session_state.api_base_url,
                message=prompt,
                session_id=st.session_state.session_id,
            )

        if response_data:
            assistant_message = response_data.get("message", "No response received.")
            metadata = response_data.get("metadata", {})
            returned_session = response_data.get(
                "session_id", st.session_state.session_id
            )
            st.session_state.session_id = returned_session

            # Save assistant message to session state
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message,
                    "metadata": metadata,
                }
            )

            # Rerun to render new message with full formatting and HITL buttons
            st.rerun()
