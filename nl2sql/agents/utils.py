"""NL2SQL agent utilities.

This module provides helper functions used across agent nodes:
  - Formatting conversation histories for LLM prompts
  - Validating SQL syntax non-destructively via PostgreSQL PREPARE statements
  - Executing SQL queries safely through SQLAlchemy engines
  - Formatting raw tabular database results into compact strings for LLM analysis
"""

import pandas as pd
from langchain_core.messages import BaseMessage
from loguru import logger
from sqlalchemy import text

from nl2sql.agents.state import State
from nl2sql.database.base import SQLBaseConnector


def get_chat_history(messages: list[BaseMessage], last_n_messages: int = 10) -> str:
    """Format the recent conversation history into a clean string for LLM prompts.
    
    Why this is needed:
      LLMs have finite context windows and can get distracted by lengthy, irrelevant
      conversations from hours ago. By slicing the last `N` messages (default: 10),
      we retain immediate context (e.g., pronouns like "Show me their orders")
      while keeping prompt latency and token costs low.

    Args:
        messages: List of LangChain BaseMessage objects (HumanMessage, AIMessage).
        last_n_messages: Maximum number of recent messages to include.

    Returns:
        str: Formatted multi-line string:
             "HUMAN: Show top 5 customers"
             "AI: Here is the query..."
    """
    return "\n".join(
        [f"{msg.type.upper()}: {msg.content}" for msg in messages[-last_n_messages:]]
    )


def validate_sql_syntax(query: str, db_conn: SQLBaseConnector) -> dict:
    """Validate SQL query syntax against PostgreSQL without actually executing it.
    
    How this works:
      PostgreSQL's `PREPARE statement_name AS <query>` command forces the database engine
      to parse the query, verify syntax, and check that referenced tables and columns exist.
      Crucially, it does NOT execute the query or return table rows!
      
      Immediately afterwards, `DEALLOCATE statement_name` frees the prepared statement
      from database session memory so we do not leak resources.

    Args:
        query: The SQL statement to validate.
        db_conn: A SQLBaseConnector instance connected to PostgreSQL.

    Returns:
        dict: {
            "valid_syntax": bool,  # True if syntax & schema are valid, False otherwise
            "error": str or None   # Database error message if syntax validation failed
        }
    """
    # Obtain a raw cursor directly from the underlying connection pool
    cur = db_conn.engine.raw_connection().cursor()
    result = {"valid_syntax": True, "error": None}

    try:
        # Prepare the query to trigger Postgres query planner & syntax parser
        cur.execute(f"PREPARE validate_stmt AS {query}")
        # Clean up the prepared statement from the session cache
        cur.execute("DEALLOCATE validate_stmt")
    except Exception as e:
        # If Postgres reports any error (e.g. syntax error, unknown column), capture it
        result["valid_syntax"] = False
        result["error"] = str(e)
    finally:
        # Ensure cursor is always closed even if exceptions occur
        cur.close()

    return result


def format_answer(state: State) -> str:
    """Format the generated SQL query and explanation for presentation to the user.

    Args:
        state: Current graph state containing `sql_query` and optional `sql_explanation`.

    Returns:
        str: Markdown-formatted string with a syntax-highlighted SQL block.
    """
    formatted_answer = f"**SQL:**\n```sql\n{state.sql_query}\n```"

    if state.sql_explanation:
        formatted_answer += f"\n\n**Explanation:**\n{state.sql_explanation}\n"

    return formatted_answer


def execute_sql_query(query: str, db_conn: SQLBaseConnector) -> dict:
    """Execute a SQL query against PostgreSQL and return tabular records and metrics.
    
    Safety & Resource Management:
      - Uses SQLAlchemy connection context manager (`with db_conn.engine.connect() as conn:`)
        to guarantee connections are checked back into the pool after execution.
      - Uses `sqlalchemy.text()` to safely wrap the raw query string.
      - Checks `query_result.returns_rows` to distinguish SELECT queries from commands.
      - Converts rows to a list of dicts (`records`) for easy JSON serialization.

    Args:
        query: The SQL query to execute.
        db_conn: A SQLBaseConnector instance.

    Returns:
        dict: {
            "success": bool,            # True if executed without error
            "data": list[dict] or None, # Query records (e.g. [{"id": 1, "name": "foo"}])
            "row_count": int,           # Number of rows returned
            "error": str or None,       # Error message if execution failed
            "query_executed": str       # Cleaned SQL query string
        }
    """
    result = {
        "success": False,
        "data": None,
        "row_count": 0,
        "error": None,
        "query_executed": query.strip(),
    }

    logger.debug(
        f"🔄 Executing SQL query: {query[:100]}{'...' if len(query) > 100 else ''}"
    )

    try:
        # Check out a connection from the SQLAlchemy connection pool
        with db_conn.engine.connect() as conn:
            query_result = conn.execute(text(query))
            logger.debug(f"Returns rows: {query_result.returns_rows}")

            # If the query produced data rows (e.g. SELECT statement)
            if query_result.returns_rows:
                # Convert results to DataFrame then to record dictionaries
                result["data"] = pd.DataFrame(
                    data=query_result.fetchall(), columns=query_result.keys()
                ).to_dict("records")
                result["row_count"] = len(result["data"])
                logger.debug(
                    "✅ Query executed successfully. "
                    f"Returned {result['row_count']} rows."
                )

            result["success"] = True

    except Exception as e:
        # Capture database runtime errors without crashing the agent
        result["error"] = str(e)
        result["success"] = False
        logger.error(f"❌ SQL execution failed: {e}")

    return result


def format_query_results_for_llm(
    execution_result: dict, max_rows: int = 5, max_cols: int | None = None
) -> str:
    """Format SQL execution results into a compact, human-readable summary for LLM interpretation.
    
    Why this is needed:
      Database queries can return thousands of rows. Feeding thousands of rows into an LLM
      would cause context overflow, high latency, and expensive token bills.
      This function provides a summary:
        1. Total rows & columns count
        2. Column names
        3. A sample preview (first `max_rows` rows)
        4. Truncation indicators if data was trimmed

    Args:
        execution_result: Result dictionary returned by `execute_sql_query`.
        max_rows: Maximum sample rows to display in the prompt (default: 5).
        max_cols: Maximum columns to display in the prompt (default: all).

    Returns:
        str: Clean text representation suitable for LLM reasoning.
    """
    # 1. Handle query failure case
    if not execution_result["success"]:
        return f"Query execution failed: {execution_result['error']}"

    # 2. Handle empty result case (e.g. zero rows matched filter)
    if execution_result["data"] is None or len(execution_result["data"]) == 0:
        logger.warning("Query executed successfully but returned no results.")
        return "Query executed successfully but returned no results (0 rows)."

    # 3. Convert records into a Pandas DataFrame for neat tabular formatting
    df = pd.DataFrame(execution_result["data"])

    # Header with overall shape and column names
    result_text = (
        "Query executed successfully. "
        f"Returned {len(df)} rows and {len(df.columns)} columns.\n"
        f"Columns: {', '.join(df.columns)}\n\n"
    )

    # 4. Add preview of the top rows
    result_text += "Results Sample:\n"
    result_text += df.iloc[:max_rows, :max_cols].to_string(index=False)

    # 5. Note if rows or columns were truncated
    if len(df) > max_rows:
        result_text += f"\n\n... ({len(df) - max_rows} more rows not shown)"
    if max_cols is not None and len(df.columns) > max_cols:
        result_text += f"\n\n... ({len(df.columns) - max_cols} more columns not shown)"

    return result_text

