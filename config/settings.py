"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for all service connections and model parameters."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # LLM provider: "anthropic" or "local"
    llm_provider: str = "anthropic"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_max_tokens: int = 4096
    anthropic_temperature: float = 0.0

    # Local model (Ollama / vLLM / llama.cpp - OpenAI-compatible endpoint)
    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "qwen2.5"
    local_llm_max_tokens: int = 4096

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "documents"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # Retrieval
    retrieval_top_k: int = 10
    rerank_top_n: int = 5
    similarity_threshold: float = 0.5

    # FastAPI
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False

    # Logging
    log_level: str = "INFO"


settings = Settings()
