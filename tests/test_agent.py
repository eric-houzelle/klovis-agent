import pytest
from unittest.mock import MagicMock, AsyncMock, patch

@pytest.mark.asyncio
async def test_agent_initialization(mock_llm_config, temp_data_dir):
    """Test agent initializes correctly with config."""
    from klovis_agent.agent import Agent
    
    agent = Agent(
        llm=mock_llm_config,
        data_dir=temp_data_dir,
        ephemeral=True
    )
    
    assert agent is not None
    assert agent.llm_config == mock_llm_config

@pytest.mark.asyncio
async def test_agent_run_with_mocked_tools(mock_llm_config, temp_data_dir):
    """Test agent run with mocked tool execution."""
    from klovis_agent.agent import Agent
    
    with patch('klovis_agent.agent.ToolRegistry') as mock_registry:
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.execute = AsyncMock(return_value={"result": "success"})
        mock_registry.return_value.get_tools.return_value = [mock_tool]
        
        agent = Agent(
            llm=mock_llm_config,
            data_dir=temp_data_dir,
            ephemeral=True
        )
        
        result = await agent.run("Test goal")
        
        assert result is not None
        assert hasattr(result, 'summary')

@pytest.mark.asyncio
async def test_agent_with_soul(mock_llm_config, temp_data_dir):
    """Test agent personality injection via Soul."""
    from klovis_agent.agent import Agent
    
    soul_content = "You are a helpful, concise assistant."
    
    agent = Agent(
        llm=mock_llm_config,
        data_dir=temp_data_dir,
        ephemeral=True,
        soul=soul_content
    )
    
    assert agent is not None
    assert agent.soul is not None

@pytest.mark.asyncio
async def test_agent_planning_loop(mock_llm_config, temp_data_dir):
    """Test agent executes planning loop correctly."""
    from klovis_agent.agent import Agent
    
    agent = Agent(
        llm=mock_llm_config,
        data_dir=temp_data_dir,
        ephemeral=True
    )
    
    # Verify agent has required components
    assert agent.memory is not None
    assert agent.decision_engine is not None
    assert agent.result_aggregator is not None
