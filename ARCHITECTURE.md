# Architecture

Agentic RAG system for legal document retrieval and question answering, built with LangGraph and backed by Qdrant (vector search) and Neo4j (knowledge graph).

## Directory Structure

```
agentic-rag-system/
├── config/
│   ├── __init__.py
│   ├── settings.py          - Pydantic BaseSettings loading all config from .env
│   └── prompts.py           - System prompt and prompt templates for the LLM
├── src/
│   ├── __init__.py
│   ├── main.py              - FastAPI app entrypoint
│   ├── schema/
│   │   ├── __init__.py      - Re-exports all schema types
│   │   ├── state.py         - LangGraph AgentState TypedDict
│   │   ├── models.py        - Pydantic models (Chunk, Citation, QueryPlan, etc.)
│   │   └── enums.py         - Enums for query types, strategies, confidence levels
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py       - Builds and compiles the LangGraph state graph
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── parse.py     - Classifies query type and extracts key entities
│   │   │   ├── plan.py      - LLM-based retrieval planning with fallback table
│   │   │   ├── retrieve.py  - Executes the chosen retrieval tools
│   │   │   ├── evaluate.py  - Decides if enough context or more exploration needed
│   │   │   ├── resolve.py   - Explores discovered sections and resolves cross-refs
│   │   │   ├── synthesize.py - Combines retrieved chunks into a draft answer
│   │   │   └── respond.py   - Formats final answer with citations
│   │   └── edges.py         - Conditional routing logic between nodes
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── vector_search.py       - Semantic search over Qdrant
│   │   ├── graph_query.py         - Cypher queries against Neo4j
│   │   ├── cross_reference.py     - Follows section cross-references
│   │   ├── sub_question.py        - Decomposes complex queries into sub-questions
│   │   ├── hierarchical_lookup.py - Navigates document section hierarchy
│   │   └── propositional_search.py - Searches over propositional index
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── strategy.py      - Maps query types to retrieval strategies
│   │   ├── qdrant_client.py - Qdrant connection manager, search, and upsert
│   │   ├── neo4j_client.py  - Neo4j driver wrapper with Cypher helpers
│   │   └── reranker.py      - Reranks retrieved chunks by relevance
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pipeline.py      - Orchestrates the full ingestion flow
│   │   ├── parser.py        - Parses raw documents (PDF, text) into sections
│   │   ├── chunker.py       - Splits sections into overlapping chunks
│   │   ├── embedder.py      - Generates embeddings via HuggingFace models
│   │   ├── qdrant_loader.py - Loads embedded chunks into Qdrant
│   │   ├── graph_builder.py - Builds Neo4j nodes and relationships from sections
│   │   └── metadata.py      - Extracts and attaches metadata to chunks
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py        - Anthropic API client wrapper
│   │   └── query_gen.py     - LLM-based query generation for sub-questions
│   └── utils/
│       ├── __init__.py
│       ├── citations.py     - Builds citation objects from retrieved context
│       ├── references.py    - Parses section reference strings (e.g., "31.020(a)")
│       └── logging.py       - JSON-formatted structured logging with correlation IDs
├── tests/
│   ├── __init__.py
│   ├── conftest.py          - Shared fixtures for tests
│   ├── test_graph/
│   │   ├── __init__.py
│   │   ├── test_nodes.py    - Unit tests for graph node functions
│   │   └── test_edges.py    - Unit tests for edge routing logic
│   ├── test_tools/
│   │   ├── __init__.py
│   │   └── test_vector_search.py - Tests for vector search tool
│   ├── test_retrieval/
│   │   ├── __init__.py
│   │   └── test_strategy.py - Tests for strategy selection
│   └── test_ingestion/
│       ├── __init__.py
│       └── test_chunker.py  - Tests for document chunking
├── scripts/
│   ├── ingest.py            - CLI script to run the ingestion pipeline
│   └── query.py             - CLI script to run queries against the system
├── app.py                   - Streamlit frontend (chat, ingestion, debug views)
├── pyproject.toml           - Project metadata, dependencies, tool config
├── docker-compose.yml       - Qdrant and Neo4j services
├── .env.example             - Template for environment variables
├── .gitignore               - Standard Python gitignore
└── ARCHITECTURE.md          - This file
```

## Data Flow

1. **Ingestion**: Documents go through `parser -> chunker -> embedder -> qdrant_loader` and `parser -> graph_builder -> Neo4j`. Metadata is attached at each stage.

2. **Query processing**: User query enters the LangGraph state machine:
   - `parse` - classify query type, extract entities
   - `plan` - LLM picks retrieval tools (falls back to a static table on failure)
   - `retrieve` - call the chosen tools (vector search, graph query, etc.)
   - `evaluate` - LLM decides if enough context exists or if more sections need exploring
   - `resolve` - explores discovered sections via graph/hierarchy, resolves cross-refs
   - The evaluate/resolve loop repeats until the LLM is satisfied or max iterations hit
   - `synthesize` - combine chunks into a draft answer
   - `respond` - format with citations and return

3. **Tools**: Each tool wraps a specific retrieval pattern. The graph nodes decide which tools to call based on the query plan.

## Key Dependencies

- **LangGraph** - state machine for the agentic loop
- **Qdrant** - vector similarity search
- **Neo4j** - knowledge graph for entity relationships and document structure
- **Anthropic Claude** - LLM for query classification, synthesis, and response generation
- **HuggingFace** - local embedding model (BGE-small)
- **FastAPI** - HTTP API layer
