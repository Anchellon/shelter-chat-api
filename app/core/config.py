from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server
    port: int = 3000

    # Ollama (shared base URL)
    ollama_base_url: str = "http://localhost:11434"

    # Per-role LLM config: provider is one of "ollama", "openai", "anthropic"
    classifier_provider: str = "ollama"
    classifier_model: str = "qwen2.5:7b"

    intake_provider: str = "ollama"
    intake_model: str = "qwen2.5:7b"

    formatter_provider: str = "ollama"
    formatter_model: str = "qwen2.5:7b"

    # Provider API keys (only needed when provider is set to that service)
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # MCP server
    mcp_server_url: str = "http://localhost:8001/mcp"

    # PostgreSQL (for AsyncPostgresSaver checkpointer)
    database_url: str = "postgresql://postgres:mypassword@localhost:5432/sheltertech"

    # Auth — if empty list, auth is disabled (local dev mode)
    api_key_header: str = "X-API-Key"
    api_keys: list[str] = []

    # LangSmith observability (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "shelter-chat"


settings = Settings()
