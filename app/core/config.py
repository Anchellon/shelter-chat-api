from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server
    port: int = 3000

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"

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
