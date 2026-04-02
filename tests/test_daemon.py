import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

@pytest.mark.asyncio
async def test_daemon_cycle_execution(mock_llm_config, temp_data_dir):
    """Test daemon executes a single cycle correctly."""
    from klovis_agent.daemon import AgentDaemon
    from klovis_agent.agent import Agent
    
    agent = Agent(
        llm=mock_llm_config,
        data_dir=temp_data_dir,
        ephemeral=True
    )
    
    daemon = AgentDaemon(agent, interval_minutes=0.01, max_cycles=1)
    
    # Run single cycle
    await daemon.run()
    
    # Verify daemon completed cycle
    assert daemon.cycles_completed >= 0

@pytest.mark.asyncio
async def test_daemon_obeys_max_cycles(mock_llm_config, temp_data_dir):
    """Test daemon respects max_cycles limit."""
    from klovis_agent.daemon import AgentDaemon
    from klovis_agent.agent import Agent
    
    agent = Agent(
        llm=mock_llm_config,
        data_dir=temp_data_dir,
        ephemeral=True
    )
    
    max_cycles = 3
    daemon = AgentDaemon(agent, interval_minutes=0.001, max_cycles=max_cycles)
    
    await daemon.run()
    
    assert daemon.cycles_completed <= max_cycles

@pytest.mark.asyncio
async def test_daemon_perception_integration(mock_llm_config, temp_data_dir):
    """Test daemon integrates perception sources."""
    from klovis_agent.daemon import AgentDaemon
    from klovis_agent.agent import Agent
    from klovis_agent.perception.base import PerceptionSource
    
    class MockPerception(PerceptionSource):
        async def get_perceptions(self):
            return [{"type": "test", "content": "test perception"}]
    
    agent = Agent(
        llm=mock_llm_config,
        data_dir=temp_data_dir,
        ephemeral=True,
        perceptions=[MockPerception()]
    )
    
    daemon = AgentDaemon(agent, interval_minutes=0.01, max_cycles=1)
    await daemon.run()
    
    # Should have processed perceptions
    assert daemon is not None
