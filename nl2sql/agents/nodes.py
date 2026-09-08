# ==============================================================================
# NL2SQL AGENT NODES & ROUTERS
# ==============================================================================
# This module defines the individual processing units (nodes) and decision gates
# (routers) that form the LangGraph workflow.
#
# In LangGraph:
#   - A "Node" is a Python function: (state: State, ...) -> dict
#     It receives the current state, performs logic (e.g. calling an LLM or DB),
#     and returns a dictionary of state updates which LangGraph merges into State.
#   - A "Router" is a conditional edge function: (state: State) -> str
#     It examines the updated state and returns a string key determining which
#     branch the workflow should follow next.
# ==============================================================================

from typing import Literal

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt
from loguru import logger

from nl2sql.agents.state import State
from nl2sql.agents.utils import (
    execute_sql_query,
    format_answer,
    format_query_results_for_llm,
    get_chat_history,
    validate_sql_syntax,
)
from nl2sql.config import UNSAFE_SQL_KEYWORDS, load_chat_prompt_template
from nl2sql.database.postgresql import PostgreSQLConnector
from nl2sql.knowledge_base.data_dictionary import DataDictionary
from nl2sql.knowledge_base.sql_examples import SQLExample
from nl2sql.knowledge_base.vector_store import VectorStore
from nl2sql.llm import get_chat_model

# ==============================================================================
# Global Prompts & Knowledge Base Loading
# ==============================================================================
# Pre-load prompt templates from YAML to avoid file I/O on every node invocation.
intent_classifier_prompt = load_chat_prompt_template(target_prompt="intent_classifier")
sql_generator_prompt = load_chat_prompt_template(target_prompt="sql_generator")

# Pre-load the Data Dictionary (contains all table structures, relationships, and descriptions)
# and SQL Examples (few-shot training examples for query synthesis).
data_dictionary = DataDictionary.load()
sql_examples = SQLExample.from_yaml("knowledge/sql_examples.yml")


# ==============================================================================
# Agent Nodes
# ==============================================================================


def intent_classifier(state: State) -> dict:
    """Classify whether the user's message is conversational chat or a database query.
    
    Why this is needed:
        Users may ask conversational questions ("What tables exist?", "Can you explain this query?")
        or direct SQL retrieval requests ("Show top 10 customers by revenue"). Routing
        to the appropriate specialized handler saves LLM tokens and prevents unnecessary
        database execution workflows.

    Args:
        state: Current graph state containing conversation history.

    Returns:
        dict: State update with `user_intent` ("sql" or "chat") and `user_query`.
    """
    logger.info("🔄 [Node] Intent Classifier")

    # 1. Extract conversation context
    # We take all messages up to the latest one as history, and the last message as the current prompt.
    chat_history = get_chat_history(state.messages[:-1])
    user_query = state.messages[-1].content
    logger.debug(f"Chat history:\n{chat_history}")
    logger.debug(f"Last message: {user_query}")

    # 2. Invoke fast, low-latency LLM with intent classification prompt
    # Temperature=0 ensures deterministic, consistent classification decisions.
    llm = get_chat_model(model_type="fast", temperature=0)
    router_llm_chain = intent_classifier_prompt | llm

    response = router_llm_chain.invoke(
        {"user_message": user_query, "chat_history": chat_history}
    )

    detected_user_intent = response.content.strip().lower()
    logger.debug(f"Detected user intent: {detected_user_intent}")

    # 3. Defensive fallback
    # If the LLM generates anything unexpected, default to conversational chat
    # rather than triggering SQL generation on arbitrary text.
    if detected_user_intent not in ["chat", "sql"]:
        logger.warning(
            "⚠️ LLM Router returned unexpected classification: "
            f"'{detected_user_intent}'. Defaulting to chat."
        )
        return {"user_intent": "chat"}

    return {"user_intent": detected_user_intent, "user_query": user_query}


def chat_agent(state: State, vector_store: VectorStore) -> dict:
    """Handle conversational and informational queries using a ReAct-style agent.
    
    Equips the model with tools to:
      - Search similar query examples in the vector store
      - Explain SQL queries step-by-step
      - Inspect database schema details
    
    Args:
        state: Current graph state with message history.
        vector_store: Vector store used by chat tools for semantic similarity search.

    Returns:
        dict: State update with the assistant's response message added to `messages`.
    """
    logger.info("🔄 [Node] Chat Agent")

    # Extract user input and previous conversational history
    messages = state.messages
    last_user_message = messages[-1]
    chat_history = messages[:-1]

    # --------------------------------------------------------------------------
    # Initialize ReAct Agent Tools
    # --------------------------------------------------------------------------
    # Import locally to keep node startup lightweight
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import StructuredTool

    from nl2sql.agents.tools.chat_tools import ChatAgentTools

    # Wrap ChatAgentTools methods into LangChain StructuredTools
    chat_agent_tools = ChatAgentTools(vector_store)
    tools = [
        StructuredTool.from_function(func=chat_agent_tools.get_similar_queries),
        StructuredTool.from_function(func=chat_agent_tools.explain_query),
        StructuredTool.from_function(func=chat_agent_tools.get_schema_info),
    ]

    # System prompt instructing the agent on its role and when to use tools
    chat_system_prompt = """You are a helpful NL2SQL assistant that can help users with database queries and data analysis.

You have access to tools that can:
1. Find similar SQL query examples from the knowledge base
2. Explain SQL queries in simple terms
3. Provide database schema information

For general questions about capabilities, explain what you can do.
For questions about the database or SQL, use the appropriate tools to provide helpful information.
Be conversational and helpful, and guide users toward useful database queries."""  # noqa: E501

    # Prompt template containing history placeholders and agent scratchpad for tool calling
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", chat_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Initialize fast chat model
    llm = get_chat_model(model_type="fast", temperature=0)

    # Construct the OpenAI tool-calling agent and wrap in an AgentExecutor
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Invoke the agent with conversational input and history
    try:
        response = agent_executor.invoke(
            {"input": last_user_message.content, "chat_history": chat_history}
        )
        response_content = response["output"]
    except Exception as e:
        # Graceful fallback if tool execution or LLM call fails
        logger.error(f"Chat agent error: {e}")
        response_content = (
            "I'm here to help you with database queries and data analysis. "
            "What would you like to explore in your database?"
        )

    logger.debug(f"✅ Chat Agent response: {response_content[:50]}...")
    return {"messages": [AIMessage(content=response_content)]}


def sql_generator(state: State, vector_store: VectorStore) -> dict:
    """Generate a SQL query from natural language using RAG and structured LLM output.
    
    RAG (Retrieval-Augmented Generation) Strategy:
      1. Schema Context: Complete database table structures, foreign keys, and column comments.
      2. Few-shot Examples: Top-4 semantically similar question-to-SQL examples retrieved
         from the vector database using vector similarity search.
      3. Structured Output: Instructs the LLM to return valid JSON containing the SQL query
         and an explanation.

    Args:
        state: Current graph state containing the user query and message history.
        vector_store: Vector store used for semantic similarity search on SQL examples.

    Returns:
        dict: Parsed dictionary from LLM containing `sql_query` and `sql_explanation`.
    """
    logger.info("🔄 [Node] SQL Generator")
    logger.debug(f"User question: {state.user_query}")

    # 1. Retrieve chat history for conversational awareness (e.g. follow-up questions)
    chat_history = get_chat_history(state.messages[:-1])
    logger.debug(f"Chat history length: {len(chat_history)}")

    # 2. Load schema context (table definitions, primary/foreign keys, descriptions)
    schema_context = data_dictionary.format_context()

    # 3. Retrieve relevant few-shot examples via vector similarity search
    retrieved_docs = vector_store.vectorstore.similarity_search(
        state.user_query, k=4, filter={"type": "example"}
    )
    logger.debug(f"Retrieved {len(retrieved_docs)} docs")
    sql_examples_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 4. Initialize LLM with structured JSON output
    # `method="json_mode"` enforces that the model responds in parseable JSON format.
    llm = get_chat_model(
        model_type="fast", temperature=0
    ).with_structured_output(method="json_mode")  # returns dict directly

    # 5. Build and invoke the generation chain
    llm_chain = sql_generator_prompt | llm

    response = llm_chain.invoke(
        {
            "user_query": state.user_query,
            "chat_history": chat_history,
            "schema_context": schema_context,
            "sql_examples": sql_examples_context,
        }
    )

    # TODO: Add key validation for the response

    return response


def sql_safety_validator(state: State) -> dict:
    """Validate that the generated SQL query is safe and strictly read-only.
    
    Security Design:
      To prevent accidental data loss, malicious SQL injection, or unauthorized database mutations,
      we check the query against a blacklist of forbidden SQL commands (`UNSAFE_SQL_KEYWORDS`).
      Using regex word boundaries (`\bkeyword\b`) ensures keywords like `UPDATE` or `DROP` are matched
      without false positives on column names like `updated_at` or `backdrop`.

    Args:
        state: Current graph state containing the generated `sql_query`.

    Returns:
        dict: Updated `sql_safety_status` ("safe" or "unsafe") and an error message if unsafe.
    """
    import re

    logger.info("🔄 [Node] SQL Safety Validator")

    # Scan for forbidden keywords using regex word boundary matching
    found_unsafe_keywords = []
    for keyword in UNSAFE_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", state.sql_query, re.IGNORECASE):
            found_unsafe_keywords.append(keyword.upper())

    # If any destructive keywords were found, halt the pipeline
    if found_unsafe_keywords:
        logger.error(f"❌ SQL contains unsafe keywords: {found_unsafe_keywords}")
        ai_message = AIMessage(
            content=f"❌ SQL query contains unsafe modification keywords: {found_unsafe_keywords}. "
                    "Only read-only SELECT queries are allowed."
        )
        return {"messages": [ai_message], "sql_safety_status": "unsafe"}
    else:
        logger.debug("✅ SQL is safe")
        return {"sql_safety_status": "safe"}


def sql_syntax_validator(state: State, db_connector: PostgreSQLConnector) -> dict:
    """Validate SQL syntax against PostgreSQL with an automated LLM self-healing loop.
    
    Validation Strategy:
      1. Tests the query syntax by running `PREPARE stmt AS <query>` on PostgreSQL.
         PostgreSQL parses the query and checks table/column existence without executing rows.
      2. Self-Correction Loop (Self-Healing):
         If PostgreSQL returns a syntax or schema error (e.g. column does not exist),
         the error message along with the broken query is fed into a reasoning LLM (`sql_syntax_fixer`).
         The LLM analyzes the error and produces a corrected query.
      3. Retries up to `max_retries` (3 times) before giving up.

    Args:
        state: Current graph state containing `sql_query`.
        db_connector: Database connector used for syntax checking via PREPARE.

    Returns:
        dict: State update with `sql_syntax_status` ("valid" or False) and repaired `sql_query`.
    """
    logger.info("🔄 [Node] SQL Syntax Validator")

    max_retries = 3
    current_query = state.sql_query

    for attempt in range(max_retries):
        # 1. Validate the current query against PostgreSQL parser
        validation_result = validate_sql_syntax(current_query, db_connector)

        # 2. If valid, return immediately
        if validation_result["valid_syntax"]:
            logger.debug("✅ Syntax Validator: SQL syntax is valid.")
            return {
                "sql_syntax_status": "valid",
                "sql_query": current_query,
            }

        # 3. If invalid, enter self-repair using the reasoning model
        logger.debug(f"Invalid syntax → Entering Fix Attempt #{attempt + 1}...")
        logger.debug(f"Current query: {current_query}")
        logger.debug(f"Error: {validation_result['error']}")

        # Load syntax repair prompt and invoke reasoning LLM
        sql_syntax_fixer_prompt = load_chat_prompt_template(
            target_prompt="sql_syntax_fixer"
        )
        llm = get_chat_model(model_type="powerful", temperature=0)
        llm_chain = sql_syntax_fixer_prompt | llm

        fixed_query = llm_chain.invoke(
            {
                "query": current_query,
                "error": validation_result["error"],
            }
        ).content

        # Update candidate query for the next validation attempt
        current_query = fixed_query

    # All retry attempts exhausted
    logger.warning(
        f"⚠️ Syntax Validator: Failed to fix SQL syntax after {max_retries} attempts."
    )
    return {
        "sql_syntax_status": False,
        "messages": [
            AIMessage(
                content=(
                    f"❌ I couldn't generate valid SQL syntax. Error: "
                    f"{validation_result['error']}"
                )
            )
        ],
    }


def human_feedback(state: State) -> dict:
    """Pause graph execution using LangGraph's interrupt to ask for human confirmation (HITL).
    
    Human-in-the-Loop (HITL) Workflow:
      1. Formats the generated SQL and its explanation into a confirmation message.
      2. Invokes LangGraph's `interrupt(...)`.
         This pauses execution and saves the current graph state checkpoint into persistent storage (PostgresSaver).
      3. The API server returns the confirmation prompt to the user.
      4. When the user responds with "yes" or "no", the API server calls `graph.invoke(Command(resume=...))`,
         which resumes execution right at this point with the user's reply.
      5. Parses the user's response into "approved" or "rejected".

    Args:
        state: Current graph state containing `sql_query` and `sql_explanation`.

    Returns:
        dict: State update with `user_feedback_status` ("approved" or "rejected")
              and recorded messages.
    """
    logger.info("🔄 [Node] Human Feedback")

    # 1. Format the proposed SQL and explanation for user review
    formatted_answer = format_answer(state)
    formatted_answer += "\nShould I execute this query? Answer with 'yes' or 'no'."

    # 2. Prepare the AI confirmation message
    ai_message = AIMessage(content=formatted_answer)

    # 3. Interrupt execution and wait for human response
    # LangGraph checkpoints the current state to the database and pauses.
    human_reply = interrupt(
        {
            "messages": ai_message,
            "waiting_for_confirmation": True,
        }
    )

    # 4. Extract and normalize the human reply when resumed
    if isinstance(human_reply, str):
        human_reply_content = human_reply.strip().lower()
    elif hasattr(human_reply, "content"):
        human_reply_content = human_reply.content.strip().lower()
    else:
        human_reply_content = str(human_reply).strip().lower()

    # Check if the user confirmed (looks for 'y' in response, e.g. "yes", "yeah", "y")
    user_feedback_status = "approved" if "y" in human_reply_content else "rejected"

    if user_feedback_status == "approved":
        logger.debug(f"✅ Human feedback: {user_feedback_status}")
    else:
        logger.debug(f"❌ Human feedback: {user_feedback_status}")

    # Record the user's reply message in the conversation history
    human_message = HumanMessage(content=human_reply_content)

    return {
        "messages": [ai_message, human_message],
        "user_feedback_status": user_feedback_status,
    }


def sql_executor(state: State, db_connector: PostgreSQLConnector) -> dict:
    """Execute the approved SQL query against the PostgreSQL database.
    
    Args:
        state: Current graph state containing the validated and approved `sql_query`.
        db_connector: PostgreSQL connector with connection engine.

    Returns:
        dict: State update with `sql_execution_status` ("success" or "failure")
              and `sql_execution_result` (containing records, row counts, or errors).
    """
    logger.info("🔄 [Node] SQL Executor")

    # Execute the query and capture formatted results / errors
    sql_execution_result = execute_sql_query(state.sql_query, db_connector)
    sql_execution_status = "success" if sql_execution_result["success"] else "failure"

    if sql_execution_status == "success":
        logger.debug(f"✅ SQL execution status: {sql_execution_status}")
        logger.debug(f"✅ SQL result: {str(sql_execution_result['data'])[:50]}...")
    else:
        logger.debug(f"❌ SQL execution status: {sql_execution_status}")
        logger.debug(f"❌ SQL result: {sql_execution_result['error']}")

    return {
        "sql_execution_status": sql_execution_status,
        "sql_execution_result": sql_execution_result,
    }


def sql_result_analyzer(state: State) -> dict:
    """Synthesize raw SQL execution data into a plain-English explanation and business insight.
    
    Why this is needed:
        Raw database records (e.g. arrays of numbers, IDs, and dates) can be difficult
        for users to interpret. This node uses a powerful LLM to summarize the key findings,
        spot trends, and directly answer the user's initial question in natural language.

    Args:
        state: Current graph state containing `user_query`, `sql_query`, and `sql_execution_result`.

    Returns:
        dict: State update with the final analyzed text in `sql_execution_analysis` and `messages`.
    """
    logger.info("🔄 [Node] SQL Result Analyser")

    # 1. Convert result records back to a Pandas DataFrame for formatting
    execution_result_for_formatting = state.sql_execution_result.copy()
    if execution_result_for_formatting["data"] is not None:
        execution_result_for_formatting["data"] = pd.DataFrame(
            execution_result_for_formatting["data"]
        )

    # 2. Format results (sample rows, column summaries) safely within token limits
    formatted_results = format_query_results_for_llm(execution_result_for_formatting)

    # 3. Load prompt template for result interpretation
    result_analyzer_prompt = load_chat_prompt_template(target_prompt="result_analyzer")

    # 4. Initialize reasoning model (slight temperature 0.1 for fluent text synthesis)
    llm = get_chat_model(model_type="powerful", temperature=0.1)

    # 5. Invoke the interpretation chain
    llm_chain = result_analyzer_prompt | llm

    response = llm_chain.invoke(
        {
            "user_query": state.user_query,
            "sql_query": state.sql_query,
            "query_results": formatted_results,
        }
    )

    analyzed_result = response.content
    logger.info("✅ Results analyzed successfully")

    # 6. Wrap result in an AIMessage and return to state
    final_message = AIMessage(content=analyzed_result)

    return {
        "messages": [final_message],
        "sql_execution_analysis": analyzed_result,
    }


# ==============================================================================
# Node Routers (Conditional Edge Logic)
# ==============================================================================
# In LangGraph, router functions read the updated State and return a string key.
# This string key maps to the next node according to the mapping dictionary
# defined in `add_conditional_edges(...)` inside `graph.py`.


def route_intent(state: State) -> Literal["sql", "chat"]:
    """Route after `intent_classifier`: sends to `sql_generator` or `chat_agent`."""
    logger.debug(f"→ Routing to {state.user_intent}")
    return state.user_intent


def check_sql_generation(state: State) -> Literal["success", "failure"]:
    """Route after `sql_generator`: verifies non-empty SQL query was generated."""
    if state.sql_query and state.sql_query.strip():
        logger.debug("→ Routing to success")
        return "success"
    else:
        logger.debug("→ Routing to failure")
        return "failure"


def check_sql_safety(state: State) -> Literal["safe", "unsafe"]:
    """Route after `sql_safety_validator`: verifies query contains no destructive keywords."""
    logger.debug(f"→ Routing to {state.sql_safety_status}")
    return "safe" if state.sql_safety_status == "safe" else "unsafe"


def check_sql_syntax(state: State) -> Literal["valid", "invalid"]:
    """Route after `sql_syntax_validator`: verifies query passed database syntax validation."""
    logger.debug(f"→ Routing to {state.sql_syntax_status}")
    return "valid" if state.sql_syntax_status == "valid" else "invalid"


def check_human_feedback(state: State) -> Literal["approved", "rejected"]:
    """Route after `human_feedback`: checks if user gave permission to execute query."""
    logger.debug(f"→ Routing to {state.user_feedback_status}")
    return "approved" if state.user_feedback_status == "approved" else "rejected"


def check_sql_execution(state: State) -> Literal["success", "failure"]:
    """Route after `sql_executor`: checks if database execution succeeded."""
    logger.debug(f"→ Routing to {state.sql_execution_status}")
    return "success" if state.sql_execution_status == "success" else "failure"

