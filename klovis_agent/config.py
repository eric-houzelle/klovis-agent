from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from klovis_agent.llm.types import ModelRoutingPolicy


class SandboxConfig(BaseModel):
    """Sandbox configuration. Set backend to 'local' or 'opensandbox'."""

    backend: str = "local"

    timeout: int = 30
    max_output_bytes: int = 1_000_000

    domain: str = "localhost:8080"
    api_key: str = ""
    protocol: str = "http"
    image: str = "opensandbox/code-interpreter:v1.0.2"
    timeout_minutes: int = 10
    python_version: str = "3.11"
    keep_alive: bool = False


class LLMConfig(BaseModel):
    """LLM provider connection configuration."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    default_model: str = "gpt-4o"
    max_tokens: int = 4096
    temperature: float = 0.2
    routing_policy: ModelRoutingPolicy = Field(
        default_factory=lambda: ModelRoutingPolicy(execution_max_tokens=16384)
    )


class AgentConfig(BaseSettings):
    """Centralized agent configuration, loaded from environment variables."""

    model_config = {"env_prefix": "AGENT_", "env_nested_delimiter": "__"}

    llm: LLMConfig = Field(default_factory=LLMConfig)
    max_iterations: int = 25
    verbose: bool = False

    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    data_dir: str = ""
    cache_dir: str = ""

    db_url: str = ""
