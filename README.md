# agentic-rag-system

RAG system for legal/regulatory documents. Built on LangGraph with Qdrant for vector search and Neo4j for the knowledge graph. Supports both Claude API and local Qwen models.

The idea is pretty simple: legal documents are full of cross-references and nested hierarchies. A basic RAG setup misses most of that structure. So this system classifies your question, picks the right retrieval strategy (vector, graph, or both), follows cross-references automatically, and synthesizes an answer with proper citations.

## How it works

```
User question
     |
   parse  -->  classify query type (definitional, procedural, structural, compliance, temporal)
     |
   plan   -->  pick primary + fallback retrieval strategies
     |
 retrieve -->  call the right tools (vector search, graph query, hierarchical lookup, etc.)
     |
 resolve  -->  follow any cross-references found in the results (loops if needed)
     |
synthesize -> combine everything into an answer with confidence level
     |
  respond  --> format with citations and return
```

The whole thing is a LangGraph state machine with conditional edges. If retrieval comes back sparse, it automatically tries secondary strategies. If chunks reference other sections, it resolves those before synthesizing.

## Setup

### 1. Start the databases

```bash
docker compose up -d
```

This gives you Qdrant on `localhost:6333` and Neo4j on `localhost:7687`.

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your keys. Two LLM options:

**Option A - Claude API:**
```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

**Option B - Local Qwen (via Ollama):**
```
LLM_PROVIDER=local
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=qwen2.5
```

If you're using Ollama, pull the model first: `ollama pull qwen2.5`

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

### 4. Ingest some documents

Drop your PDFs/HTML/text files in a folder and run:

```bash
python scripts/ingest.py --input-dir /path/to/docs --collection legal_docs
```

This parses the documents, chunks them (preserving section hierarchy), generates embeddings, loads everything into Qdrant, and builds the knowledge graph in Neo4j.

### 5. Run it

**Streamlit UI** (recommended for trying it out):
```bash
streamlit run app.py
```

**FastAPI server** (for integration):
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

**One-off query from terminal:**
```bash
python scripts/query.py --question "Can an applicant appeal a denied permit?"
```

## The 6 retrieval tools

| Tool | What it does | When it's used |
|------|-------------|----------------|
| `vector_search` | Semantic similarity search over Qdrant | Most queries - the default starting point |
| `graph_query` | Traverses entities and relationships in Neo4j | Structural questions ("how do X and Y relate?") |
| `cross_reference` | Resolves references like "see Section 31.020(a)(1)" | Automatically when chunks contain cross-refs |
| `sub_question` | Breaks complex multi-part questions into simpler ones | Complex queries that need parallel retrieval |
| `hierarchical_lookup` | Navigates parent/child/sibling sections | Finding context around a specific section |
| `propositional_search` | Searches by fact type (rule, exception, penalty, condition) | Compliance questions |

## Strategy selection

The system picks a strategy based on what kind of question you asked:

| Query type | Primary | Fallbacks |
|-----------|---------|-----------|
| Definitional ("What is X?") | Vector search | Graph query, Hierarchical |
| Procedural ("How do I do X?") | Vector search | Graph query |
| Structural ("How do A and B relate?") | Graph query | Vector search |
| Compliance ("Am I violating X?") | Propositional search | Graph query |
| Temporal ("When does X take effect?") | Vector search | Hierarchical |

## Project structure

```
config/           - settings, prompts
src/schema/       - state definitions, pydantic models, enums
src/graph/        - LangGraph nodes, edges, builder
src/tools/        - the 6 retrieval tools
src/retrieval/    - Qdrant/Neo4j clients, strategy selector, reranker
src/ingestion/    - document parsing, chunking, embedding, loading
src/llm/          - LLM client (Claude + local), query expansion
src/utils/        - citations, reference parsing, logging
tests/            - 54 tests with mocked DB fixtures
scripts/          - CLI for ingestion and querying
app.py            - Streamlit frontend
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full file-by-file breakdown.

## Tests

```bash
pytest
```

All tests use mocked Qdrant/Neo4j fixtures, so you don't need running databases to run them.

## Tech stack

- **LangGraph** - agentic loop and state management
- **LlamaIndex** - document parsing and embedding
- **Qdrant** - vector database
- **Neo4j** - graph database
- **Anthropic Claude / Qwen** - LLM (switchable)
- **FastAPI** - API server
- **Streamlit** - frontend
- **BGE-small-en-v1.5** - embedding model (runs locally)
