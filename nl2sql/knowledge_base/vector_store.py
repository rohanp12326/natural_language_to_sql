"""Vector-based document storage and retrieval for RAG workflows.

This module provides the `VectorStore` interface built on top of `langchain_postgres.PGVector`.
It indexes two distinct knowledge domains into PostgreSQL:
  1. Database Schema Documents (`type: "schema"`):
     - Table definitions, column descriptions, data types, primary & foreign keys.
     - Enables the agent to locate the right tables for any user inquiry.
  2. SQL Examples (`type: "example"`):
     - Golden question-to-SQL pairs for few-shot prompt enrichment.
     - Helps the LLM learn complex joins, date functions, and query patterns.
"""

import hashlib

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_postgres import PGVector

from nl2sql.database.base import SQLBaseConnector
from nl2sql.knowledge_base.data_dictionary import DataDictionary
from nl2sql.knowledge_base.sql_examples import SQLExample
from nl2sql.llm import get_embeddings


class VectorStore:
    """PGVector store for indexing and retrieving NL2SQL schema and example documents."""

    def __init__(
        self,
        db_connector: SQLBaseConnector,
        collection_name: str = "nl2sql_embeddings",
        embeddings: Embeddings | None = None,
    ) -> None:
        """Initialize PGVector collection using the application database connection.

        Args:
            db_connector: Database connector whose SQLAlchemy engine connects to PostgreSQL.
            collection_name: Name of the vector collection table in PostgreSQL (default: "nl2sql_embeddings").
            embeddings: LangChain Embeddings instance. Defaults to local ONNX embeddings.
        """
        self.vectorstore = PGVector(
            connection=db_connector.engine,
            embeddings=embeddings or get_embeddings(),
            collection_name=collection_name,
        )

    def as_retriever(self, **kwargs):
        """Expose standard LangChain retriever interface for vector similarity search."""
        return self.vectorstore.as_retriever(**kwargs)

    def add_documents(self, documents: list[Document]) -> None:
        """Index documents into the vector store with deterministic IDs for idempotency.

        Idempotency Strategy:
          To prevent duplicate embeddings when running indexing scripts multiple times:
          - Schema documents use their qualified table name (`schema.table`) as ID.
          - Example documents use an MD5 hash of their unique title as ID.
          Existing documents with matching IDs are updated rather than duplicated.

        Args:
            documents: List of LangChain Document objects to vectorize and store.
        """
        ids = [
            f"{doc.metadata['schema']}.{doc.metadata['table']}"
            if doc.metadata.get("type") == "schema"
            else hashlib.md5(doc.metadata["title"].encode("utf-8")).hexdigest()
            for doc in documents
        ]
        self.vectorstore.add_documents(documents, ids=ids)

    def get_documents_from_data_dictionary(
        self,
        data_dictionary: DataDictionary,
    ) -> list[Document]:
        """Convert a hierarchical DataDictionary into flat Document objects for vector search.

        Traverses: Database -> Schema -> Table.
        Each table becomes a single Document where:
          - `page_content`: Formatted Markdown with table name, comments, columns, and foreign keys.
          - `metadata`: Contains filterable tags (`type: "schema"`, schema name, table name, PKs, FKs).

        Args:
            data_dictionary: Loaded DataDictionary instance.

        Returns:
            list[Document]: Documents ready to be vectorized.
        """
        documents = []

        for database_name, database_info in data_dictionary.databases.items():
            for schema_name, schema_info in database_info.schemas.items():
                for table_name, table_info in schema_info.tables.items():
                    # Format table structure and columns into context string
                    content = table_info.format_context()

                    # Attach searchable metadata for structured filtering
                    metadata = {
                        "type": "schema",
                        "database": database_name,
                        "schema": schema_name,
                        "table": table_name,
                        "primary_keys": ", ".join(table_info.primary_keys),
                        "foreign_keys": ", ".join(
                            col
                            for fk in table_info.foreign_keys
                            for col in fk["constrained_columns"]
                        ),
                    }

                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)

        return documents

    def get_documents_from_sql_examples(
        self,
        sql_examples: dict[str, SQLExample],
    ) -> list[Document]:
        """Convert a dictionary of SQLExample instances into Document objects for few-shot search.

        Args:
            sql_examples: Mapping of example title -> SQLExample.

        Returns:
            list[Document]: Documents with question/SQL content and metadata `type: "example"`.
        """
        documents = []

        for title, example in sql_examples.items():
            doc_content = example.format_context()
            doc_metadata = {"type": "example", "title": title}
            documents.append(Document(page_content=doc_content, metadata=doc_metadata))

        return documents

