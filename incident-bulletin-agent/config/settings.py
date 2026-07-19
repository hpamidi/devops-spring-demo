from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    llm_provider: Literal["anthropic", "openai"] = Field(default="anthropic", alias="LLM_PROVIDER")
    llm_model: str = Field(default="claude-sonnet-4-6", alias="LLM_MODEL")

    # Azure AI Search
    azure_search_endpoint: str = Field(default="", alias="AZURE_SEARCH_ENDPOINT")
    azure_search_key: str = Field(default="", alias="AZURE_SEARCH_KEY")
    azure_search_index: str = Field(default="historical-bulletins", alias="AZURE_SEARCH_INDEX")

    # External services
    vcc_base_url: str = Field(default="https://vcc.internal/api/v1", alias="VCC_BASE_URL")
    vcc_api_key: str = Field(default="", alias="VCC_API_KEY")
    bcu_base_url: str = Field(default="https://bcu.internal/api/v1", alias="BCU_BASE_URL")
    bcu_api_key: str = Field(default="", alias="BCU_API_KEY")

    # LangSmith
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="incident-bulletin-agent", alias="LANGCHAIN_PROJECT")

    # Prometheus
    prometheus_port: int = Field(default=8001, alias="PROMETHEUS_PORT")

    # App
    app_env: Literal["development", "staging", "production"] = Field(
        default="development", alias="APP_ENV"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    http_timeout: int = Field(default=30, alias="HTTP_TIMEOUT")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    templates_dir: Path = Field(default=Path("templates"), alias="TEMPLATES_DIR")


@lru_cache
def get_settings() -> Settings:
    return Settings()
