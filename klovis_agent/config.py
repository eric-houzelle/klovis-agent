from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from klovis_agent.llm.types import ModelRoutingPolicy


class SandboxConfig(BaseModel):
    """Sandbox configuration. Set backend to 'local' or 'opensandbox'."""

    backend: Literal["local", "opensandbox"] = Field(
        default="local",
        description="Sandbox backend type: 'local' for local execution, 'opensandbox' for remote."
    )

    timeout: int = Field(
        default=30,
        description="Timeout in seconds for sandbox execution."
    )
    max_output_bytes: int = Field(
        default=1_000_000,
        description="Maximum output size in bytes."
    )

    domain: str = Field(
        default="localhost:8080",
        description="Domain for opensandbox backend."
    )
    api_key: Optional[str] = Field(
        default="",
        description="API key for opensandbox authentication."
    )
    protocol: Literal["http", "https"] = Field(
        default="http",
        description="Protocol for opensandbox connection."
    )
    image: str = Field(
        default="opensandbox/code-interpreter:v1.0.2",
        description="Docker image for opensandbox."
    )
    timeout_minutes: int = Field(
        default=10,
        description="Timeout in minutes for opensandbox session."
    )
    python_version: str = Field(
        default="3.11",
        description="Python version for opensandbox environment."
    )
    keep_alive: bool = Field(
        default=False,
        description="Whether to keep the sandbox alive after execution."
    )


class LLMConfig(BaseModel):
    """LLM provider connection configuration."""

    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the LLM API."
    )
    api_key: Optional[str] = Field(
        default="",
        description="API key for LLM authentication."
    )
    default_model: str = Field(
        default="gpt-4o",
        description="Default model to use for LLM calls."
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for LLM responses."
    )
    temperature: float = Field(
        default=0.2,
        description="Temperature for LLM sampling (0.0-1.0)."
    )
    routing_policy: ModelRoutingPolicy = Field(
        default_factory=lambda: ModelRoutingPolicy(execution_max_tokens=16384),
        description="Policy for routing requests to different models."
    )


class AgentConfig(BaseSettings):
    """Centralized agent configuration, loaded from environment variables."""

    model_config = {"env_prefix": "AGENT_", "env_nested_delimiter": "__"}

    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration settings."
    )
    max_iterations: int = Field(
        default=25,
        description="Maximum iterations for agent execution loop."
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging output."
    )

    sandbox: SandboxConfig = Field(
        default_factory=SandboxConfig,
        description="Sandbox configuration settings."
    )

    data_dir: Optional[str] = Field(
        default="",
        description="Directory for agent data storage."
    )
    cache_dir: Optional[str] = Field(
        default="",
        description="Directory for agent cache storage."
    )

    db_url: Optional[str] = Field(
        default="",
        description="Database connection URL for persistent storage."
    )
