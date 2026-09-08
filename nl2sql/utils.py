"""Package-level utility helpers for the NL2SQL project.

Provides:
  - CLI section header formatting for clean terminal output.
  - Standalone SQL syntax validator using PostgreSQL PREPARE statements.
"""

from nl2sql.database.base import SQLBaseConnector


def print_section(title: str, separator: str = "=") -> None:
    """Print a visually prominent section header in terminal logs.
    
    Example:
        print_section("Database Migration")
        # Database Migration
        # ==================
    """
    print(f"{title}\n{separator * len(title)}")


def validate_sql_syntax(query: str, db_conn: SQLBaseConnector) -> dict:
    """Validate SQL syntax and column existence using PostgreSQL's PREPARE statement.

    How it works:
      1. Uses `PREPARE stmt AS <query>`: Forces PostgreSQL parser and planner to inspect
         the query for syntax validity and catalog existence without executing table rows.
      2. Uses `DEALLOCATE stmt`: Frees the prepared statement from the session cache immediately.

    Args:
        query: The SQL query to validate.
        db_conn: A SQLBaseConnector instance.

    Returns:
        dict: {
            "valid_syntax": bool,  # True if syntax is valid, False otherwise
            "error": str or None   # Database error description if invalid
        }
    """
    cur = db_conn.engine.raw_connection().cursor()
    result = {"valid_syntax": True, "error": None}

    try:
        cur.execute(f"PREPARE validate_stmt AS {query}")
        cur.execute("DEALLOCATE validate_stmt")
    except Exception as e:
        result["valid_syntax"] = False
        result["error"] = str(e)
    finally:
        cur.close()

    return result

