import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for test isolation."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)

@pytest.fixture
def mock_llm_config():
    """Create a mock LLMConfig for testing."""
    from klovis_agent.config import LLMConfig
    return LLMConfig(api_key="test-key", default_model="gpt-4o-mini")

@pytest.fixture
def mock_llm_response():
    """Fixture for mocking LLM responses."""
    async def mock_generate(*args, **kwargs):
        return MagicMock(
            content="Mocked response",
            usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )
    return mock_generate

@pytest.fixture
def agent_kwargs(temp_data_dir):
    """Common kwargs for creating test agents."""
    return {
        "data_dir": temp_data_dir,
        "ephemeral": True,
    }

@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    tool = MagicMock()
    tool.name = "mock_tool"
    tool.description = "A mock tool for testing"
    tool.execute = AsyncMock(return_value={"success": True, "result": "mocked"})
    return tool

@pytest.fixture(autouse=True)
def patch_external_calls():
    """Patch external API calls to prevent real requests during tests."""
    with patch('klovis_agent.llm.openai_llm.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        yield mock_client
