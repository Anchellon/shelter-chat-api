from langchain_core.language_models.chat_models import BaseChatModel


def get_llm(provider: str, model: str, json_mode: bool = False, max_tokens: int | None = None) -> BaseChatModel:
    """
    Return a LangChain chat model for the given provider and model name.
    json_mode=True forces JSON output (supported by ollama and openai).
    """
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        from app.core.config import settings
        kwargs = dict(base_url=settings.ollama_base_url, model=model, temperature=0)
        if json_mode:
            kwargs["format"] = "json"
        if max_tokens is not None:
            kwargs["num_predict"] = max_tokens
        return ChatOllama(**kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        from app.core.config import settings
        kwargs = dict(model=model, temperature=0, api_key=settings.openai_api_key)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        from app.core.config import settings
        kwargs = dict(model=model, temperature=0, api_key=settings.anthropic_api_key)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatAnthropic(**kwargs)

    raise ValueError(f"Unknown LLM provider: {provider!r}. Expected 'ollama', 'openai', or 'anthropic'.")
