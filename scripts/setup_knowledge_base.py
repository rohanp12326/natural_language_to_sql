"""Setup Complete Knowledge Base (Data Dictionary + PGVector Store).

This bootstrapping script prepares the knowledge retrieval layer for the NL2SQL agent:
  1. Data Dictionary Generation:
     - Introspects live PostgreSQL database tables, columns, foreign keys, and descriptions.
     - Saves the structured data catalog to `knowledge/data_dictionary.yml`.
  2. Vector Store Ingestion:
     - Embeds all table schema documentation into PGVector (`nl2sql_embeddings`).
     - Loads few-shot SQL examples from `knowledge/sql_examples.yml` and embeds them
       into PGVector with metadata `type: example`.
"""

from loguru import logger

from nl2sql.config import load_schema_config
from nl2sql.database import PostgreSQLConnector
from nl2sql.knowledge_base.data_dictionary import DataDictionary
from nl2sql.knowledge_base.sql_examples import SQLExample
from nl2sql.knowledge_base.vector_store import VectorStore

# Filepaths for serialized knowledge assets
DATA_DICTIONARY_OUTPUT_PATH = "knowledge/data_dictionary.yml"
SQL_EXAMPLES_PATH = "knowledge/sql_examples.yml"


def setup_data_dictionary(
    db_connector: PostgreSQLConnector, schema: dict
) -> DataDictionary:
    """Extract schema metadata from PostgreSQL and serialize to YAML.

    Args:
        db_connector: PostgreSQL connector with active connection engine.
        schema: Target schema/table configuration mapping.

    Returns:
        DataDictionary: Populated in-memory data dictionary.
    """
    logger.info("Extracting data dictionary from database...")
    data_dictionary = DataDictionary.from_inspector(
        inspector=db_connector.inspector,
        database_schema=schema,
    )

    # Diagnostic check: notify developers if tables or columns lack comments
    missing_tables = data_dictionary.get_tables_with_missing_descriptions()
    missing_columns = data_dictionary.get_columns_with_missing_descriptions()

    if missing_tables:
        logger.warning(f"There are tables with missing descriptions: {missing_tables}")
    if missing_columns:
        logger.warning(
            f"There are columns with missing descriptions: {missing_columns}"
        )

    # Save to YAML so nodes can load without re-introspecting the database on startup
    logger.info(f"Saving data dictionary to {DATA_DICTIONARY_OUTPUT_PATH}...")
    data_dictionary.save(DATA_DICTIONARY_OUTPUT_PATH)
    logger.success("Data dictionary setup complete.")

    return data_dictionary


def setup_vector_store(
    db_connector: PostgreSQLConnector, data_dictionary: DataDictionary
) -> None:
    """Vectorize schema definitions and SQL examples into PostgreSQL PGVector.

    Args:
        db_connector: PostgreSQL connector.
        data_dictionary: Extracted data dictionary containing table structures.
    """
    logger.info("Initializing vector store...")
    vector_store = VectorStore(db_connector)

    # Convert table structures to LangChain Documents and insert into PGVector
    logger.info("Adding schema documents to vector store...")
    vector_store.add_documents(
        vector_store.get_documents_from_data_dictionary(data_dictionary)
    )

    # Load few-shot SQL examples and insert into PGVector
    logger.info("Loading SQL examples...")
    sql_examples = SQLExample.from_yaml(SQL_EXAMPLES_PATH)

    logger.info("Adding SQL example documents to vector store...")
    vector_store.add_documents(
        vector_store.get_documents_from_sql_examples(sql_examples)
    )
    logger.success("Vector store setup complete.")


if __name__ == "__main__":
    logger.info("Starting knowledge base setup...")

    # Load target database configuration and schema scope
    logger.info("Loading schema config...")
    schema = load_schema_config()

    logger.info("Connecting to database...")
    db_connector = PostgreSQLConnector(config_path="configs/database.yml")

    # Step 1: Introspect and generate data dictionary YAML
    logger.info("Step 1/2: Setting up data dictionary...")
    data_dictionary = setup_data_dictionary(db_connector, schema)

    # Step 2: Ingest schema and SQL examples into PGVector
    logger.info("Step 2/2: Setting up vector store...")
    setup_vector_store(db_connector, data_dictionary)

    logger.success("Complete knowledge base setup finished! 🎉")

