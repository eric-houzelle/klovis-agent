import pytest
from unittest.mock import patch, AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_full_agent_run_with_mocked_llm(mock_llm_config, temp_data_dir, mock_llm_response):
    """Integration test: full agent run with mocked LLM."""
    from klovis_agent.agent import Agent
    
    with patch('klovis_agent.llm.openai_llm.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Task complete"))]
        mock_client.chat.completions.create = MagicMock(return_value=mock_completion)
        mock_openai.return_value = mock_client
        
        agent = Agent(
            llm=mock_llm_config,
            data_dir=temp_data_dir,
            ephemeral=True
        )
        
        result = await agent.run("Write a test file")
        
        assert result is not None
        assert result.summary is not None

@pytest.mark.asyncio
async def test_agent_with_custom_tools(mock_llm_config, temp_data_dir):
    """Integration test: agent with custom injectable tools."""
    from klovis_agent.agent import Agent
    from klovis_agent.tools.base import BaseTool
    
    class CustomTool(BaseTool):
        name = "custom_tool"
        description = "A custom test tool"
        
        async def execute(self, **kwargs):
            return {"custom": "result", "value": 42}
    
    agent = Agent(
        llm=mock_llm_config,
        data_dir=temp_data_dir,
        ephemeral=True,
        tools=[CustomTool()]
    )
    
    # Verify tool is registered
    tools = agent.tool_registry.get_tools()
    assert any(t.name == "custom_tool" for t in tools)

@pytest.mark.asyncio
async def test_agent_memory_persistence_across_runs(mock_llm_config, temp_data_dir):
    """Integration test: memory persists across agent runs."""
    from klovis_agent.agent import Agent
    
    # First run - store memory
    agent1 = Agent(
        llm=mock_llm_config,
        data_dir=temp_data_dir,
        ephemeral=False
    )
    
    await agent1.memory.store_episodic("First run completed", tags=["test"])
    await agent1.close()
    
    # Second run - recall memory
    agent2 = Agent(
        llm=mock_llm_config,
        data_dir=temp_data_dir,
        ephemeral=False
    )
    
    memories = await agent2.memory.recall("First run", zone="episodic")
    
    assert len(memories) >= 1
    await agent2.close()
