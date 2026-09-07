"""PostgreSQL Database Connector.

Implements `SQLBaseConnector` for PostgreSQL using modern `psycopg` (psycopg 3)
and SQLAlchemy 2.0.

Provides:
  - SQLAlchemy Engine creation (for ORM, inspector reflection, and PGVector)
  - Native psycopg Connection generation (required by LangGraph's PostgresSaver checkpointer)
"""

from pathlib import Path
from urllib.parse import unquote_plus

from sqlalchemy import Engine, create_engine

from nl2sql.database.base import DatabaseParams, SQLBaseConnector


class PostgreSQLConnector(SQLBaseConnector):
    """Specialized database connector for PostgreSQL instances."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Initialize PostgreSQL connector and inherit parameter resolution."""
        super().__init__(host, port, database, username, password, config_path)

    def create_uri(
        self,
        params: DatabaseParams,
        dialect: str = "postgresql",
        driver: str = "psycopg",
    ) -> str:
        """Create a SQLAlchemy-compatible database connection URI.

        Format:
            postgresql+psycopg://username:password@host:port/database

        Args:
            params: Resolved connection credentials and host settings.
            dialect: SQL dialect name (default: "postgresql").
            driver: DBAPI driver name (default: "psycopg" for psycopg v3).

        Returns:
            str: Connection string formatted for `sqlalchemy.create_engine()`.
        """
        dialect_driver = f"{dialect}+{driver}" if driver else dialect
        return f"{dialect_driver}://{params.username}:{params.password}@{params.host}:{params.port}/{params.database}"

    def create_postgresql_uri(self, params: DatabaseParams | None = None) -> str:
        """Create a standard PostgreSQL libpq connection URI.
        
        Why this is needed:
          Native drivers like `psycopg.connect()` expect the standard URI scheme
          without the '+psycopg' dialect driver prefix:
          `postgresql://username:password@host:port/database`.
          `unquote_plus` decodes percent-encoded characters because psycopg accepts
          raw string passwords directly.

        Args:
            params: DatabaseParams to use. If None, uses `self.params`.

        Returns:
            str: Standard PostgreSQL URI string.
        """
        if params is None:
            params = self.params

        password = unquote_plus(params.password)
        return f"postgresql://{params.username}:{password}@{params.host}:{params.port}/{params.database}"

    def get_psycopg_connection(
        self,
        autocommit: bool = True,
        prepare_threshold: int = 0,
        row_factory: object = None,
        **kwargs: str | int | bool,
    ) -> object:
        """Create a raw psycopg v3 connection object.
        
        Why this is needed:
          LangGraph's state persistence layer (`PostgresSaver`) requires a direct
          psycopg connection object rather than a SQLAlchemy connection pool.

        Args:
            autocommit: If True, executes queries outside explicit transactions.
            prepare_threshold: Number of times a query runs before preparing server-side.
            row_factory: Custom row converter (e.g. `dict_row` for dictionary access).
            **kwargs: Additional driver options forwarded to psycopg.

        Returns:
            psycopg.Connection: Active PostgreSQL connection.
        """
        from psycopg import Connection

        conn_string = self.create_postgresql_uri()
        return Connection.connect(
            conn_string,
            autocommit=autocommit,
            prepare_threshold=prepare_threshold,
            row_factory=row_factory,
            **kwargs,
        )

    def create_engine(self, params: DatabaseParams) -> Engine:
        """Create a SQLAlchemy connection Engine using the psycopg driver."""
        return create_engine(self.create_uri(params))

