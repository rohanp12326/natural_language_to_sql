Listed directory SQL-db-with-natural-language
Listed directory nl2sql-agent
Listed directory olist-db
Viewed README.md:1-89
Viewed README.md:1-132
Listed directory nl2sql
Listed directory agents
Viewed graph.py:1-103
Viewed nodes.py:1-100
Viewed nodes.py:150-260
Viewed nodes.py:261-413
Listed directory api
Listed directory notebooks

### 💡 The Big Picture (The 10-Second Summary)

Imagine you have a company database with millions of rows of sales and customer records. Normally, only someone who knows **SQL** (database query language) can extract insights from it.

This project is an **AI Assistant that lets anyone talk to a database in plain English**. 
Instead of writing complex SQL queries like:
```sql
SELECT c.customer_city, SUM(p.payment_value) 
FROM orders o JOIN ...
```
You can simply type:
> *"Which cities had the highest sales revenue last quarter?"*

The AI translates your words into an accurate SQL query, runs it securely on the database, and gives you a clear, human explanation of the results.

---

### 📂 The Two Main Parts of This Repository

This project is split into two complementary folders:

1. **[`olist-db`](file:///run/media/rohanp12326/Local_Disc_4/2026%20Learnings/projects/SQL-db-with-natural-language/olist-db)**: **The Database (The Knowledge Bank)**
   - A containerized **PostgreSQL** database pre-loaded with over **1.4 million real-world e-commerce records** (from Olist, a major Brazilian e-commerce platform).
   - Contains data on orders, customers, product categories, reviews, payments, and marketing leads.
   - Includes **PGVector** (vector database extension) for semantic similarity searches.

2. **[`nl2sql-agent`](file:///run/media/rohanp12326/Local_Disc_4/2026%20Learnings/projects/SQL-db-with-natural-language/nl2sql-agent)**: **The AI Agent (The Brain)**
   - Built using **LangGraph** and **LangChain** (featured at the SciPy 2025 conference).
   - Implements an intelligent, multi-step agent workflow with safety checks, self-healing query repair, and human approvals.

---

### 🔄 How It Works: The Step-by-Step Flow

The workflow is orchestrated as a state machine in [graph.py](file:///run/media/rohanp12326/Local_Disc_4/2026%20Learnings/projects/SQL-db-with-natural-language/nl2sql-agent/nl2sql/agents/graph.py) and [nodes.py](file:///run/media/rohanp12326/Local_Disc_4/2026%20Learnings/projects/SQL-db-with-natural-language/nl2sql-agent/nl2sql/agents/nodes.py). Here is what happens when you ask a question:

```
                  ┌────────────────────────┐
                  │   User Question        │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ 1. Intent Classifier   │
                  └─────┬────────────┬─────┘
          "General Chat"│            │"Database Question"
                        │            │
                        ▼            ▼
             ┌──────────────┐   ┌───────────────────────────┐
             │ Chat Agent   │   │ 2. SQL Generator (RAG)    │
             └──────────────┘   └─────────────┬─────────────┘
                                              │
                                              ▼
                                ┌───────────────────────────┐
                                │ 3. SQL Safety Validator   │
                                └─────────────┬─────────────┘
                                              │ (Safe)
                                              ▼
                                ┌───────────────────────────┐
                                │ 4. Syntax Validator & Fix │
                                └─────────────┬─────────────┘
                                              │ (Valid)
                                              ▼
                                ┌───────────────────────────┐
                                │ 5. Human-in-the-Loop      │
                                └─────────────┬─────────────┘
                                              │ (Approved)
                                              ▼
                                ┌───────────────────────────┐
                                │ 6. SQL Executor           │
                                └─────────────┬─────────────┘
                                              │
                                              ▼
                                ┌───────────────────────────┐
                                │ 7. SQL Result Analyzer    │
                                └─────────────┬─────────────┘
                                              │
                                              ▼
                                     Final Answer to User
```

1. **Intent Classification (`intent_classifier`)**:
   - The AI first determines what you want: Are you just chatting (*"What can you do?"*), or asking a data question (*"Show me the top 5 products"*).
   - If it's general conversation, it routes you to the **Chat Agent**.

2. **Smart SQL Generation with RAG (`sql_generator`)**:
   - It searches a vector database (**PGVector**) to find similar past SQL questions and retrieves relevant table schema information.
   - Using this context, the LLM crafts the exact SQL query needed.

3. **Safety Guardrails (`sql_safety_validator`)**:
   - Before doing anything with the database, it inspects the generated SQL for dangerous keywords (like `DROP`, `DELETE`, `UPDATE`, `ALTER`, `TRUNCATE`).
   - If the query is unsafe, execution stops immediately to protect the database.

4. **Self-Healing Syntax Check (`sql_syntax_validator`)**:
   - Instead of blindly running the query, it tests the query syntax with PostgreSQL (`EXPLAIN`).
   - If there is an error (e.g. a misspelled column name), the agent automatically attempts to fix the query (up to 3 retries).

5. **Human Approval (`human_feedback`)**:
   - For complete peace of mind, the agent can pause and show you the query: *"Should I execute this query? Answer yes/no."*

6. **Execution (`sql_executor`)**:
   - Once confirmed, the query is executed against the live database, and raw tabular data is fetched.

7. **Result Translation (`sql_result_analyzer`)**:
   - The AI looks at the data returned and converts the numbers and rows into an easy-to-read summary with natural English explanations.

---

### 🛠️ Key Technologies Used

| Layer | Tools | What it does |
|---|---|---|
| **Agent Framework** | [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain) | Coordinates the step-by-step decision flow and agent memory |
| **LLMs** | OpenAI (GPT-4o / GPT-4.1) | Understands user queries, generates SQL, and analyzes results |
| **Database & Vector Store** | PostgreSQL + `pgvector` | Houses the 1.4M+ e-commerce dataset and stores embeddings for similarity search |
| **Web API & Deployment** | FastAPI, Docker | Provides REST endpoints and allows running everything in reproducible containers |
| **Package Manager** | `uv` | Fast Python package management |

---

### 🌟 Why This Project Stands Out

Most basic AI text-to-SQL projects just send your question to ChatGPT and hope the generated SQL works. This project is **production-ready** because:
- **It protects the database**: Read-only validation ensures no accidental deletions.
- **It can fix its own bugs**: It tests queries before running them and repairs errors.
- **It learns from examples**: Uses RAG to find similar SQL queries to improve accuracy.
- **Interactive walkthroughs**: Includes step-by-step Jupyter notebooks in the [notebooks](file:///run/media/rohanp12326/Local_Disc_4/2026%20Learnings/projects/SQL-db-with-natural-language/nl2sql-agent/notebooks) directory to learn every component hands-on.