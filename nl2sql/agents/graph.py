"""NL2SQL agent graph."""

from functools import partial

from langgraph.graph import END, START, StateGraph

from nl2sql.agents import nodes
from nl2sql.agents.state import State
from nl2sql.database.postgresql import PostgreSQLConnector
from nl2sql.knowledge_base.vector_store import VectorStore

# ==============================================================================
# NL2SQL AGENT WORKFLOW GRAPH DEFINITION
# ==============================================================================
# This module defines the core execution graph of the NL2SQL agent using LangGraph.
# 
# The pipeline follows this high-level lifecycle:
#   1. START -> intent_classifier:
#      Determine if the user wants general conversational assistance or a database query.
#   2. Branching:
#      - If "chat": Route to `chat_agent` (handles schema questions, explanations).
#      - If "sql":  Route through the multi-stage SQL execution pipeline:
#                   a. sql_generator        -> Generates SQL from question + schema/examples
#                   b. sql_safety_validator  -> Blocks dangerous operations (DROP, DELETE, etc.)
#                   c. sql_syntax_validator  -> Tests query against DB with PREPARE & auto-fixes errors
#                   d. human_feedback       -> Pauses execution (interrupt) for user confirmation (HITL)
#                   e. sql_executor         -> Runs query on PostgreSQL and collects tabular data
#                   f. sql_result_analyzer  -> Translates raw data rows into a clear natural-language summary
#   3. END:
#      Workflow completes and returns the final state to the caller.
# ==============================================================================

def create_graph(
    db_connector: PostgreSQLConnector, vector_store: VectorStore
) -> StateGraph:
    """Create and configure the NL2SQL agent graph.
    
    Args:
        db_connector: PostgreSQL connector for validating syntax and executing queries.
        vector_store: PGVector store containing schema definitions and few-shot examples.
        
    Returns:
        StateGraph: An uncompiled LangGraph workflow configured with all nodes and edges.
    """
    # Initialize the StateGraph parameterised with our State Pydantic model.
    # Every node in this graph receives and updates this shared State object.
    workflow = StateGraph(State)

    # --------------------------------------------------------------------------
    # Dependency Injection with functools.partial
    # --------------------------------------------------------------------------
    # Nodes in LangGraph typically only accept (state: State) as their signature.
    # To pass external runtime dependencies (like database connectors and vector stores)
    # into specific nodes, we pre-bind those arguments using `partial()`.
    chat_agent_node = partial(nodes.chat_agent, vector_store=vector_store)
    sql_generator_node = partial(nodes.sql_generator, vector_store=vector_store)
    sql_syntax_validator_node = partial(
        nodes.sql_syntax_validator, db_connector=db_connector
    )
    sql_executor_node = partial(nodes.sql_executor, db_connector=db_connector)

    # --------------------------------------------------------------------------
    # Node Registration
    # --------------------------------------------------------------------------
    # Each node is a discrete processing step represented by a Python callable.
    
    # 1. Classifies user query as either "chat" or "sql"
    workflow.add_node("intent_classifier", nodes.intent_classifier)
    
    # 2. General conversation agent equipped with vector search & schema tools
    workflow.add_node("chat_agent", chat_agent_node)
    
    # 3. Uses LLM + RAG (schema context & few-shot examples) to generate SQL
    workflow.add_node("sql_generator", sql_generator_node)
    
    # 4. Security check: scans generated SQL for destructive keywords (DROP, DELETE, etc.)
    workflow.add_node("sql_safety_validator", nodes.sql_safety_validator)
    
    # 5. DB syntax check via PREPARE statement with LLM self-correction retry loop
    workflow.add_node("sql_syntax_validator", sql_syntax_validator_node)
    
    # 6. Human-in-the-loop (HITL): interrupts execution to ask user for approval
    workflow.add_node("human_feedback", nodes.human_feedback)
    
    # 7. Executes the approved query against the PostgreSQL database
    workflow.add_node("sql_executor", sql_executor_node)
    
    # 8. Converts raw database rows into plain-English answers and data insights
    workflow.add_node("sql_result_analyzer", nodes.sql_result_analyzer)

    # --------------------------------------------------------------------------
    # Edge Definitions & Conditional Routing
    # --------------------------------------------------------------------------
    # Entry point: All user requests start at the intent classifier.
    workflow.add_edge(START, "intent_classifier")

    # Step 1 Routing: Route based on intent ("sql" -> sql_generator, "chat" -> chat_agent)
    workflow.add_conditional_edges(
        "intent_classifier",
        nodes.route_intent,
        {
            "sql": "sql_generator",
            "chat": "chat_agent",
        },
    )

    # Step 2 Routing: Check if SQL generator successfully produced a query string
    workflow.add_conditional_edges(
        "sql_generator",
        nodes.check_sql_generation,
        {
            "success": "sql_safety_validator",  # Proceed to safety inspection
            "failure": END,                     # Stop if SQL generation failed
        },
    )

    # Step 3 Routing: Check if query is safe (read-only, no destructive operations)
    workflow.add_conditional_edges(
        "sql_safety_validator",
        nodes.check_sql_safety,
        {
            "safe": "sql_syntax_validator",    # Proceed to syntax validation
            "unsafe": END,                      # Abort immediately if unsafe
        },
    )

    # Step 4 Routing: Check if query has valid syntax in PostgreSQL
    workflow.add_conditional_edges(
        "sql_syntax_validator",
        nodes.check_sql_syntax,
        {
            "valid": "human_feedback",         # Proceed to human confirmation
            "invalid": END,                     # Stop if syntax couldn't be repaired
        },
    )

    # Step 5 Routing: Check if human user approved the query execution
    workflow.add_conditional_edges(
        "human_feedback",
        nodes.check_human_feedback,
        {
            "approved": "sql_executor",         # Execute query on the database
            "rejected": END,                    # Cancel execution if user declines
        },
    )

    # Step 6 Routing: Check if query execution succeeded on the database
    workflow.add_conditional_edges(
        "sql_executor",
        nodes.check_sql_execution,
        {
            "success": "sql_result_analyzer",   # Interpret query results with LLM
            "failure": END,                     # Stop if execution encountered an error
        },
    )

    # Exit points: Both conversational answers and analyzed SQL results end the workflow
    workflow.add_edge("sql_result_analyzer", END)
    workflow.add_edge("chat_agent", END)

    return workflow
