"""Configuration utilities and constants for the NL2SQL agent.

Handles:
  - Security guardrail keyword lists (blocking dangerous SQL operations)
  - Loading environment and application configs from YAML files
  - Converting YAML prompt templates into LangChain ChatPromptTemplates
"""

from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------------------------------------
# Security Guardrail: Forbidden SQL Keywords
# ------------------------------------------------------------------------------
# The NL2SQL agent is intended strictly for analytics and read-only querying (SELECT).
# Any query containing these DDL (Data Definition Language) or DML (Data Manipulation Language)
# keywords is flagged and blocked by `sql_safety_validator` to prevent data deletion or mutation.
UNSAFE_SQL_KEYWORDS = [
    "drop",       # Deletes tables, views, or databases
    "delete",     # Removes rows from a table
    "truncate",   # Empties all records in a table
    "alter",      # Modifies table schema or column constraints
    "update",     # Mutates existing row values
    "insert",     # Adds new rows to a table
    "create",     # Creates new tables, schemas, or indexes
    "rename",     # Renames existing database objects
    "grant",      # Grants security permissions to users
    "revoke",     # Revokes security permissions from users
    "deny",       # Explicitly denies permissions
]


def load_config(config_path: str | Path) -> dict:
    """Load and parse a YAML configuration file safely.

    Args:
        config_path: Path to the target YAML file.

    Returns:
        dict: Parsed YAML content as a Python dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {config_path}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {config_path}") from e


def load_database_config(config_path: str | Path = "configs/database.yml") -> dict:
    """Load database connection settings (host, port, db name, credentials).

    Args:
        config_path: Path to database YAML file (default: "configs/database.yml").

    Returns:
        dict: Nested 'database' section of the YAML file.
    """
    return load_config(config_path)["database"]


def load_schema_config(config_path: str | Path = "configs/schema.yml") -> dict:
    """Load target database schema mapping specifying which schemas and tables to index.

    Args:
        config_path: Path to schema YAML file (default: "configs/schema.yml").

    Returns:
        dict: Mapping of databases -> schemas -> list of table names.
    """
    return load_config(config_path)


def load_chat_prompt_template(
    file_path: str | Path = "configs/prompts.yml", target_prompt: str | None = None
) -> ChatPromptTemplate:
    """Load a prompt specification from a YAML file and construct a LangChain ChatPromptTemplate.

    Structure in prompts.yml:
        target_prompt:
          system_prompt: "You are an assistant..."
          user_prompt: "Question: {user_query}..."

    Args:
        file_path: Path to the prompt YAML file (default: "configs/prompts.yml").
        target_prompt: Key name identifying the prompt in the YAML file.

    Returns:
        ChatPromptTemplate: Ready-to-invoke LangChain prompt template.

    Raises:
        ValueError: If target_prompt key does not exist in the YAML file.
    """
    prompts = load_config(file_path)
    if prompts.get(target_prompt) is None:
        raise ValueError(f"Prompt template '{target_prompt}' not found in {file_path}")

    # Build a two-stage message prompt with system instructions and user input placeholders
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompts[target_prompt]["system_prompt"]),
            ("human", prompts[target_prompt]["user_prompt"]),
        ]
    )
    return prompt_template

