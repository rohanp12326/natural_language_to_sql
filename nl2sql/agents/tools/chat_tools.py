"""Chat agent toolset for conversational assistance and knowledge retrieval.

In a ReAct (Reason + Act) architecture, an LLM agent uses 'Tools' to perform actions
and gather facts beyond its internal weights. This module provides three core tools:
  1. `get_similar_queries`: Performs vector similarity search over historical SQL examples.
  2. `explain_query`: Generates an educational, step-by-step breakdown of any SQL query.
  3. `get_schema_info`: Retrieves schema, columns, types, and foreign keys from the data dictionary.
"""

from langchain_core.prompts import ChatPromptTemplate

from nl2sql.knowledge_base.data_dictionary import DataDictionary
from nl2sql.knowledge_base.vector_store import VectorStore
from nl2sql.llm import get_chat_model


class ChatAgentTools:
    """Tool provider for the conversational Chat Agent."""

    def __init__(self, vector_store: VectorStore) -> None:
        """Initialize tools with access to the vector store and data dictionary.
        
        Args:
            vector_store: VectorStore connected to PostgreSQL for similarity searches.
        """
        self.vector_store = vector_store
        
        # Load data dictionary to supply full schema context to explanation tools
        try:
            self.data_dictionary = DataDictionary.load()
            self.schema_context = self.data_dictionary.format_context()
        except Exception as e:
            # Fallback schema description in case the YAML file is not yet generated
            self.schema_context = """Database contains tables for ecommerce data including:
            - customers: customer information and locations
            - orders: order details and status
            - order_items: individual items in orders
            - products: product information and categories
            - sellers: seller information
            """  # noqa: E501
            print(f"Warning: Could not load data dictionary: {e}")

    def get_similar_queries(self, query: str, k: int = 3) -> str:
        """Retrieve K similar SQL query examples from the vector knowledge base.

        How it works:
          Converts the user's natural language question into an embedding and runs
          a cosine-distance similarity search in PostgreSQL (PGVector), filtering
          only documents tagged with metadata `type: "example"`.

        Args:
            query: The user's question (e.g. "Find top selling products").
            k: The number of similar examples to retrieve (default: 3).

        Returns:
            str: Markdown-formatted list of matching SQL queries and titles.
        """
        try:
            # Enforce a reasonable upper bound to protect prompt context
            limit = min(k, 5)
            
            # Create a retriever filtered specifically to SQL examples
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": limit, "filter": {"type": "example"}}
            )
            docs = retriever.invoke(query)

            if not docs:
                return "I couldn't find any similar queries in the knowledge base."

            # Format matching documents into clear Markdown blocks
            response_lines = []
            if k > 10:
                response_lines.append(
                    "Note: I can only retrieve a maximum of 10 examples at a time.\n"
                )

            response_lines.append(
                f"Here are the top {len(docs)} similar queries I found:"
            )

            for i, doc in enumerate(docs):
                title = doc.metadata.get("title", "SQL Example")
                sql_query = doc.page_content
                response_lines.append(
                    f"\n---\n\n**Example {i + 1}: {title}**\n\n```sql\n{sql_query}\n```"
                )

            return "\n".join(response_lines)

        except Exception as e:
            return f"Error retrieving similar queries: {e!s}"

    def explain_query(self, sql_query: str) -> str:
        """Provide a plain-English, step-by-step breakdown of a SQL query.
        
        How it works:
          Feeds the query together with the database schema context into a fast LLM.
          The schema context enables the model to explain what specific column names
          or foreign key joins actually mean in the business domain.

        Args:
            sql_query: The SQL statement to explain.

        Returns:
            str: Natural language explanation of the query's tables, filters, and aggregations.
        """
        try:
            explainer_prompt = """You are a helpful SQL expert. Explain the following SQL query in simple terms,
            considering the database schema provided.

            Database Schema:
            {schema_context}

            SQL Query to explain:
            {sql_query}

            Please provide a clear, step-by-step explanation of what this query does."""  # noqa: E501

            explainer_prompt_template = ChatPromptTemplate.from_template(
                explainer_prompt
            )
            # Use deterministic temperature for factual explanation
            llm = get_chat_model(model_type="fast", temperature=0)

            explainer_chain = explainer_prompt_template | llm

            response = explainer_chain.invoke(
                {"sql_query": sql_query, "schema_context": self.schema_context}
            )

            return response.content

        except Exception as e:
            return f"Error explaining query: {e!s}"

    def get_schema_info(self, table_name: str = "") -> str:
        """Inspect the database schema, either for a specific table or the whole database.

        Args:
            table_name: Optional name of the table to look up (e.g. "orders").
                        If empty, returns a general overview of all tables.

        Returns:
            str: Description of columns, primary keys, and relationships.
        """
        try:
            # If no specific table was requested, return overview of all tables
            if not table_name:
                return f"Here's an overview of the database schema:\n\n{self.schema_context[:1000]}..."  # noqa: E501

            # Search the schema text for the specific table section
            if table_name.lower() in self.schema_context.lower():
                lines = self.schema_context.split("\n")
                table_info = []
                capturing = False

                for line in lines:
                    if (
                        f"TABLE: {table_name}" in line
                        or f"table: {table_name}" in line.lower()
                    ):
                        capturing = True
                    elif capturing and line.startswith("TABLE:"):
                        # Stop capturing when reaching the next table definition
                        break

                    if capturing:
                        table_info.append(line)

                if table_info:
                    return f"Information about table '{table_name}':\n\n" + "\n".join(
                        table_info
                    )

            return f"Table '{table_name}' not found in the schema. Use get_schema_info() without parameters to see all available tables."  # noqa: E501

        except Exception as e:
            return f"Error retrieving schema information: {e!s}"

