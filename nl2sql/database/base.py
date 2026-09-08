"""Abstract Base Connector for SQL Databases.

This module defines the database connection interface and configuration schema:
  - `DatabaseParams`: Pydantic model with strict password redaction for secure logging.
  - `SQLBaseConnector`: Abstract Base Class (ABC) providing parameter resolution,
    SQLAlchemy engine creation, and schema inspection capabilities.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy import Engine, inspect

from nl2sql.config import load_database_config

# Load environment variables from .env file if present
load_dotenv()


class DatabaseParams(BaseModel):
    """Strongly-typed database connection parameters.
    
    Security Feature:
      `password: str = Field(repr=False)` prevents Pydantic from displaying the plain-text
      password in default object prints. The overridden `__repr__` masks passwords with asterisks,
      preventing accidental credential leaks in log files or debugging traces.
    """

    host: str
    port: int
    database: str
    username: str
    password: str = Field(repr=False)

    def __repr__(self) -> str:
        """Mask the password when converting object to string for logging or debugging."""
        return (
            f"DatabaseParams(host={self.host}, port={self.port}, "
            f"database={self.database}, username={self.username}, "
            f"password={'*' * len(self.password)})"
        )


class SQLBaseConnector(ABC):
    """Abstract base connector establishing a unified interface for SQL databases.
    
    Subclasses (e.g. `PostgreSQLConnector`) must implement dialect-specific URI formatting
    and SQLAlchemy engine instantiation.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Initialize connector, establish SQLAlchemy engine, and prepare schema inspector.
        
        Args:
            host: Database host IP or hostname.
            port: Database listening port.
            database: Target database name.
            username: User credential.
            password: User password.
            config_path: Optional path to a YAML configuration file.
        """
        # 1. Resolve parameters through priority fallback chain
        self.params = self._resolve_params(
            host, port, database, username, password, config_path
        )
        
        # 2. Initialize connection engine and schema inspector
        self.engine = self.create_engine(self.params)
        self.inspector = inspect(self.engine)

    def _resolve_params(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        config_path: Path | None = None,
    ) -> DatabaseParams:
        """Resolve connection parameters following a 3-tier fallback hierarchy:
        
        Priority:
          1. Explicit arguments passed directly to the constructor
          2. Values configured in YAML file (if `config_path` provided)
          3. Environment variables (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
        
        Special Handling:
          `quote_plus` URL-encodes special characters in passwords (e.g. '@', ':', '/')
          so they do not break SQLAlchemy URI syntax.
        """
        db_config = load_database_config(config_path) if config_path else {}

        return DatabaseParams(
            host=host or db_config.get("host") or os.getenv("DB_HOST"),
            port=port or db_config.get("port") or os.getenv("DB_PORT"),
            database=database or db_config.get("database") or os.getenv("DB_NAME"),
            username=username or db_config.get("username") or os.getenv("DB_USER"),
            password=quote_plus(
                password or db_config.get("password") or os.getenv("DB_PASSWORD")
            ),
        )

    @abstractmethod
    def create_uri(self, params: DatabaseParams, dialect: str, driver: str) -> str:
        """Construct the database-specific connection URI."""
        pass

    @abstractmethod
    def create_engine(self, params: DatabaseParams) -> Engine:
        """Create and configure a SQLAlchemy connection Engine."""
        pass

