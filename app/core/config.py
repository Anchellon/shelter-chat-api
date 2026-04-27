from pydantic import model_validator
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

    # PostgreSQL — set DATABASE_URL directly or supply individual components
    database_url: str = "postgresql://postgres:mypassword@localhost:5432/sheltertech"
    db_host: str = ""
    db_port: int = 5432
    db_name: str = "shelter"
    db_user: str = ""
    db_password: str = ""

    @model_validator(mode="after")
    def build_database_url(self) -> "Settings":
        if self.db_host and self.db_user and self.db_password:
            self.database_url = (
                f"postgresql://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        return self

    # CORS — comma-separated list of allowed origins
    cors_origins: str = "http://localhost:5173"

    # Auth0 JWT validation
    auth0_domain: str = ""
    auth0_audience: str = ""

    # LangSmith observability (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "shelter-chat"


settings = Settings()
