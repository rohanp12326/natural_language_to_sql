"""Few-Shot SQL Example Management.

This module handles example Question -> SQL query pairs:
  - Few-Shot Prompting: In-context learning technique where supplying concrete examples
    of question-to-SQL pairs in the prompt teaches the model complex joins, aggregations,
    and PostgreSQL-specific idioms (e.g. date truncation, casting).
  - Code Beautification: Uses `sqlparse` to format and indent queries consistently.
  - Serialization: Loads and saves curated examples from/to YAML (`knowledge/sql_examples.yml`).
"""

from pathlib import Path

import sqlparse
import yaml
from pydantic import BaseModel, Field


class SQLExample(BaseModel):
    """Container for a single golden Question-SQL pair used in few-shot prompting."""

    question: str = Field(..., description="Natural language user question")
    sql: str = Field(..., description="Target PostgreSQL query")

    def format_context(self) -> str:
        """Format the question and SQL query into a clean, re-indented Markdown block for prompts.

        Uses `sqlparse.format(reindent=True)` to convert single-line or messy SQL
        into clean, multi-line indented statements. This makes it easier for the LLM
        to learn structural patterns.

        Returns:
            str: "Question: ... \n```sql\nSELECT ...\n```"
        """
        formatted_sql = sqlparse.format(self.sql, reindent=True)
        return f"Question: {self.question}\n```sql\n{formatted_sql}\n```"

    @classmethod
    def from_yaml(cls, file_path: Path | str) -> dict[str, "SQLExample"]:
        """Load curated SQL examples from a YAML file.

        Args:
            file_path: Path to the examples YAML file.

        Returns:
            dict[str, SQLExample]: Dictionary mapping example keys to SQLExample instances.

        Raises:
            FileNotFoundError: If the YAML file cannot be found.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"SQL examples file not found: {file_path}")

        with open(file_path) as f:
            raw_data = yaml.safe_load(f)

        examples = {}
        for key, data in raw_data.items():
            if isinstance(data, dict):
                examples[key] = cls(**data)
            else:
                # Fallback for simple key: "SELECT..." string formats
                examples[key] = cls(question=key, sql=str(data))

        return examples

    @classmethod
    def save_yaml(
        cls, examples: dict[str, "SQLExample"], output_path: Path | str
    ) -> Path:
        """Serialize a collection of SQL examples back to a YAML file.

        Args:
            examples: Mapping of title -> SQLExample.
            output_path: Target file path.

        Returns:
            Path: Output path of the saved file.
        """
        if isinstance(output_path, str):
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Pydantic models to pure Python dictionaries for YAML dump
        data = {key: example.model_dump() for key, example in examples.items()}

        with open(output_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        return output_path

    @classmethod
    def format_all_for_prompt(cls, examples: dict[str, "SQLExample"]) -> str:
        """Combine all provided examples into a single multi-line string for static prompts."""
        return "\n\n".join(example.format_context() for example in examples.values())

