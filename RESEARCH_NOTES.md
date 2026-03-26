# Architecture Research Notes

Research performed March 2026. These notes cover current best practices for each technology in the Agentic RAG stack.

---

## 1. LangGraph StateGraph Patterns

### Package & Version
- `langgraph` 1.1.x (installed: 1.1.3)
- `langgraph-checkpoint` for persistence
- `langgraph-prebuilt` for ToolNode, tools_condition

### State Definition

State is defined as a `TypedDict` with optional `Annotated` reducers. Reducers control how node return values merge into state. Without a reducer, values are overwritten. With one (e.g., `operator.add`), values are combined.

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langchain_core.messages import BaseMessage, add_messages

class AgentState(TypedDict):
    """State flows through every node in the graph."""
    messages: Annotated[list[BaseMessage], add_messages]  # append-only chat history
    question: str                                          # current user question (overwrite)
    documents: Annotated[list[str], add]                   # accumulated retrieved docs
    generation: str                                        # final answer
    retry_count: int                                       # loop counter
```

Key import: `from langchain_core.messages import add_messages` is the standard reducer for chat message lists. It handles deduplication by message ID.

### Building the Graph

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)

# Nodes are plain functions: (state) -> partial state dict
def retrieve_node(state: AgentState) -> dict:
    docs = retriever.invoke(state["question"])
    return {"documents": docs}

def generate_node(state: AgentState) -> dict:
    answer = llm.invoke(state["documents"], state["question"])
    return {"generation": answer}

builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
```

Nodes receive the full state and return a **partial** dict with only the keys they want to update.

### Edges and Conditional Routing

```python
# Fixed edge: always go from retrieve to generate
builder.add_edge("retrieve", "generate")

# Conditional edge: route based on state
def should_retry(state: AgentState) -> str:
    if state["retry_count"] > 3:
        return END                    # stop
    if not state["documents"]:
        return "retrieve"             # try again
    return "generate"                 # proceed

builder.add_conditional_edges("grade_documents", should_retry)

# Entry point
builder.add_edge(START, "route_question")
```

`START` and `END` are sentinel constants from `langgraph.graph`.

### Prebuilt Components for Tool Calling

```python
from langgraph.prebuilt import ToolNode, tools_condition

tools = [vector_search_tool, graph_search_tool, summarize_tool]
tool_node = ToolNode(tools)

builder.add_node("tools", tool_node)
builder.add_conditional_edges(
    "agent",
    tools_condition,  # checks if last message has tool_calls
)
```

`tools_condition` returns `"tools"` if the LLM requested tool calls, or `END` if it gave a direct response.

### Compiling

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()  # in-memory; use SqliteSaver or PostgresSaver for prod
graph = builder.compile(checkpointer=checkpointer)
```

Compilation validates the graph structure (no orphaned nodes, etc.). The checkpointer enables conversation persistence and time-travel debugging.

### Invoking & Streaming

```python
# Sync invoke
result = graph.invoke({"question": "What is GDPR Article 5?"}, config={"configurable": {"thread_id": "abc123"}})

# Async streaming
async for event in graph.astream(
    {"question": "What is GDPR Article 5?"},
    config={"configurable": {"thread_id": "abc123"}},
    stream_mode=["updates", "messages"],
):
    print(event)
```

The `thread_id` in config ties the run to a checkpointed conversation.

### Emitting Progress from Nodes

```python
from langgraph.config import get_stream_writer

def retrieve_node(state: AgentState) -> dict:
    writer = get_stream_writer()
    writer({"type": "status", "message": "Searching vector database..."})
    docs = retriever.invoke(state["question"])
    writer({"type": "status", "message": f"Found {len(docs)} documents"})
    return {"documents": docs}
```

---

## 2. LlamaIndex + Qdrant Integration

### Packages

```
pip install llama-index
pip install llama-index-vector-stores-qdrant
pip install llama-index-embeddings-huggingface  # for local embeddings
pip install qdrant-client
```

### Embedding Model Setup

```python
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# BGE models are the current go-to for local embeddings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
```

Alternative with FastEmbed (lighter weight):
```python
from llama_index.embeddings.fastembed import FastEmbedEmbedding
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
```

### Qdrant Client Setup

```python
import qdrant_client

# Local instance
client = qdrant_client.QdrantClient(host="localhost", port=6333)

# In-memory (for dev/testing)
client = qdrant_client.QdrantClient(location=":memory:")

# Cloud
client = qdrant_client.QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key",
)
```

### VectorStoreIndex with Qdrant

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

vector_store = QdrantVectorStore(
    client=client,
    collection_name="legal_documents",
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# From documents (first-time ingestion)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

# From existing collection (subsequent loads)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
```

### Querying

```python
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("What are the penalties under GDPR?")
print(response.response)
print(response.source_nodes)  # retrieved chunks with metadata
```

For retriever-only use (without the built-in response synthesis):
```python
retriever = index.as_retriever(similarity_top_k=10)
nodes = retriever.retrieve("GDPR penalties")
# nodes is a list of NodeWithScore objects
```

### Async Support

```python
aclient = qdrant_client.AsyncQdrantClient(location=":memory:")
vector_store = QdrantVectorStore(
    client=client,
    aclient=aclient,        # pass both sync and async clients
    collection_name="legal_documents",
    prefer_grpc=True,       # faster for large-scale ops
)
```

### Hybrid Search (Dense + Sparse)

```python
vector_store = QdrantVectorStore(
    client=client,
    collection_name="legal_documents",
    enable_hybrid=True,
    fastembed_sparse_model="Qdrant/bm25",  # sparse model for keyword matching
)
```

This stores both dense vectors (from your embedding model) and sparse BM25 vectors. At query time, Qdrant fuses the results.

---

## 3. Neo4j Python Driver Patterns

### Package
- `neo4j` (current: 6.1.x)

### Driver Creation

```python
from neo4j import GraphDatabase, AsyncGraphDatabase

# Sync driver
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
)

# Async driver
async_driver = AsyncGraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
)
```

The driver manages a connection pool. Create it once, share across your application, close on shutdown.

### Sync Session + Managed Transactions (Preferred Pattern)

```python
def find_related_regulations(tx, regulation_id: str) -> list[dict]:
    """Transaction function: must be idempotent (driver retries on transient failures)."""
    result = tx.run(
        """
        MATCH (r:Regulation {id: $reg_id})-[:REFERENCES]->(related:Regulation)
        RETURN related.id AS id, related.title AS title, related.jurisdiction AS jurisdiction
        ORDER BY related.title
        """,
        reg_id=regulation_id,
    )
    return [dict(record) for record in result]

with driver.session(database="neo4j") as session:
    regulations = session.execute_read(find_related_regulations, "GDPR-Art5")
```

Always use `execute_read` for reads and `execute_write` for writes. The driver uses this hint for routing in cluster setups.

### Async Session + Managed Transactions

```python
async def find_related_regulations(tx, regulation_id: str) -> list[dict]:
    result = await tx.run(
        """
        MATCH (r:Regulation {id: $reg_id})-[:REFERENCES]->(related:Regulation)
        RETURN related.id AS id, related.title AS title
        """,
        reg_id=regulation_id,
    )
    return [dict(record) async for record in result]

async with async_driver.session(database="neo4j") as session:
    regulations = await session.execute_read(find_related_regulations, "GDPR-Art5")
```

### Parameterized Queries (Mandatory)

Always use `$parameter` placeholders. Never string-format Cypher queries.

```python
# CORRECT
tx.run("MATCH (n:Document {id: $doc_id}) RETURN n", doc_id="DOC-123")

# WRONG - SQL injection risk, no query caching
tx.run(f"MATCH (n:Document {{id: '{doc_id}'}}) RETURN n")
```

### Performance Best Practices

1. **Always specify the database**: `session(database="neo4j")` avoids a round-trip to discover the default.
2. **Transaction functions must be idempotent**: The driver retries them on transient failures.
3. **Sessions are not thread-safe**: Use one session per thread/coroutine.
4. **Close the driver on app shutdown**: `driver.close()` or `await async_driver.close()`.

### Useful Cypher Patterns for Legal/Regulatory Graphs

```cypher
-- Find regulatory chain (which regulations reference which)
MATCH path = (r:Regulation {id: $reg_id})-[:REFERENCES*1..3]->(target:Regulation)
RETURN path

-- Find all entities mentioned in a regulation
MATCH (r:Regulation {id: $reg_id})-[:MENTIONS]->(e:Entity)
RETURN e.name, e.type, e.description

-- Find regulations by jurisdiction and topic
MATCH (r:Regulation)-[:COVERS]->(t:Topic {name: $topic})
WHERE r.jurisdiction = $jurisdiction
RETURN r.id, r.title, r.effective_date
```

---

## 4. Hybrid Retrieval Orchestration with LangGraph

### Architecture Pattern

The recommended approach for combining vector and graph DB results in an agentic RAG system:

```
User Query
    |
    v
[Route Question]  -- decides retrieval strategy
    |
    +---> [Vector Search]  -- semantic similarity via Qdrant
    +---> [Graph Search]   -- relationship traversal via Neo4j
    +---> [Hybrid Search]  -- both, then merge
    |
    v
[Grade Documents]  -- LLM checks relevance of retrieved docs
    |
    +---> relevant  --> [Generate Answer]
    +---> not relevant --> [Rewrite Query] --> loop back
    |
    v
[Check Hallucination]  -- verify answer is grounded in docs
    |
    +---> grounded  --> END
    +---> not grounded --> [Generate Answer] (retry)
```

### Implementation as LangGraph Nodes

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)

# Route based on question type
def route_question(state: AgentState) -> str:
    """Use an LLM or heuristic to pick retrieval strategy."""
    question = state["question"]
    # Questions about relationships/connections -> graph
    # Questions about content/meaning -> vector
    # Complex questions -> both
    if needs_relationship_data(question):
        return "graph_search"
    elif needs_semantic_search(question):
        return "vector_search"
    return "hybrid_search"

builder.add_node("vector_search", vector_search_node)
builder.add_node("graph_search", graph_search_node)
builder.add_node("hybrid_search", hybrid_search_node)
builder.add_node("grade_documents", grade_documents_node)
builder.add_node("generate", generate_node)
builder.add_node("rewrite_query", rewrite_query_node)

builder.add_conditional_edges(START, route_question)
builder.add_edge("vector_search", "grade_documents")
builder.add_edge("graph_search", "grade_documents")
builder.add_edge("hybrid_search", "grade_documents")

def grade_decision(state: AgentState) -> str:
    if state["documents"]:
        return "generate"
    return "rewrite_query"

builder.add_conditional_edges("grade_documents", grade_decision)
builder.add_edge("rewrite_query", "vector_search")  # retry with rewritten query
builder.add_edge("generate", END)
```

### Merging Results from Both DBs

When doing hybrid search (vector + graph), merge and deduplicate results before grading:

```python
def hybrid_search_node(state: AgentState) -> dict:
    question = state["question"]

    # Vector search: semantic similarity
    vector_docs = qdrant_retriever.retrieve(question)

    # Graph search: relationship traversal
    graph_docs = neo4j_search(question)

    # Merge and deduplicate by document ID
    seen = set()
    merged = []
    for doc in vector_docs + graph_docs:
        doc_id = doc.metadata.get("doc_id") or doc.id_
        if doc_id not in seen:
            seen.add(doc_id)
            merged.append(doc)

    return {"documents": merged}
```

---

## 5. FastAPI Integration with LangGraph

### Basic Endpoint (Non-Streaming)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    thread_id: str | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    thread_id = req.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = await graph.ainvoke({"question": req.question}, config=config)

    return QueryResponse(
        answer=result["generation"],
        sources=[{"id": d.id_, "text": d.text[:200]} for d in result["documents"]],
    )
```

### Streaming Endpoint with Server-Sent Events (SSE)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

@app.post("/query/stream")
async def stream_query(req: QueryRequest):
    thread_id = req.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            async for event in graph.astream(
                {"question": req.question},
                config=config,
                stream_mode=["updates", "custom"],
            ):
                yield f"data: {json.dumps(event, default=str)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

### Application Lifespan (Resource Management)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize connections
    app.state.neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    app.state.qdrant_client = qdrant_client.AsyncQdrantClient(QDRANT_URL)
    yield
    # Shutdown: close connections
    await app.state.neo4j_driver.close()
    await app.state.qdrant_client.close()

app = FastAPI(lifespan=lifespan)
```

### Requirements
- Python 3.11+ (for reliable ContextVar propagation in async streaming)
- `pip install python-multipart` if accepting file uploads
- FastAPI 0.115+

---

## 6. Anthropic Claude SDK for Tool Use

### Package
- `anthropic` (Python SDK)
- Docs: https://platform.claude.com/docs/en/api/sdks/python

### Tool Definition Format

```python
tools = [
    {
        "name": "search_legal_documents",
        "description": "Search the legal document database for regulations, case law, or compliance documents matching the query. Returns relevant document excerpts with citations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what legal information to find",
                },
                "jurisdiction": {
                    "type": "string",
                    "description": "Filter by jurisdiction (e.g., 'EU', 'US', 'UK')",
                },
                "doc_type": {
                    "type": "string",
                    "enum": ["regulation", "case_law", "guidance", "all"],
                    "description": "Type of document to search for",
                },
            },
            "required": ["query"],
        },
    }
]
```

Tool names must be 1-64 alphanumeric characters (plus underscores). Descriptions should be detailed (3-4+ sentences for best results).

### Making a Tool-Use Request

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    tools=tools,
    messages=[
        {"role": "user", "content": "What are the penalties under GDPR Article 83?"}
    ],
)
```

### Processing Tool Calls and Sending Results Back

```python
def run_tool_loop(messages: list, tools: list) -> str:
    """Run the LLM with tools until it produces a final text response."""
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )

        # If the model stopped without requesting tools, we're done
        if response.stop_reason != "tool_use":
            return next(
                block.text for block in response.content if block.type == "text"
            )

        # Process each tool call
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # Execute the tool
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})
```

### Error Handling in Tool Results

```python
{
    "type": "tool_result",
    "tool_use_id": "toolu_abc123",
    "content": "Error: Document not found in database",
    "is_error": True,
}
```

Setting `is_error: True` tells the model the tool failed, so it can retry or inform the user.

### Async Client

```python
client = anthropic.AsyncAnthropic()

response = await client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    tools=tools,
    messages=messages,
)
```

### Key Design Decisions for This Project

For the agentic RAG system, Claude will NOT be the direct orchestrator via tool use. Instead:
- **LangGraph** handles orchestration (state machine, routing, retries)
- **Claude** is called within LangGraph nodes for: query understanding, document grading, answer generation, hallucination checking
- Tools are defined as LangGraph `ToolNode` entries, not as Claude tool-use schemas

However, if we want a "chat with tools" mode where Claude decides which tools to call, the tool-use loop above is the pattern. This can be wrapped as a single LangGraph node that internally runs the tool loop.

---

## Architecture Decision Summary

| Component | Role | Key Pattern |
|-----------|------|-------------|
| LangGraph | Orchestration | StateGraph with typed state, conditional routing, checkpointed persistence |
| LlamaIndex + Qdrant | Vector retrieval | VectorStoreIndex with QdrantVectorStore, BGE embeddings, hybrid search |
| Neo4j | Graph retrieval | Async driver, managed transactions, parameterized Cypher |
| Claude (Anthropic SDK) | LLM backbone | Called within LangGraph nodes for reasoning tasks |
| FastAPI | API layer | SSE streaming endpoint, async lifespan for resource management |

### Recommended Project Structure

```
agentic-rag-system/
  src/
    config/          # Settings, env vars, model configs
    schema/          # Pydantic models, state definitions
    graph/           # LangGraph nodes, edges, graph builder
    tools/           # RAG tool implementations (vector search, graph search, etc.)
    retrieval/       # Qdrant client, Neo4j client, hybrid merge logic
    ingestion/       # Document loading, chunking, embedding, indexing
    llm/             # Anthropic client wrapper, prompt templates
    api/             # FastAPI app, routes, middleware
  tests/
  scripts/           # CLI utilities, data loading scripts
  config/            # YAML/env config files
```
