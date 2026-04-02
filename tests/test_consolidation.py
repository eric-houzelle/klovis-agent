import pytest
from unittest.mock import MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_consolidation_extracts_key_facts(temp_data_dir, mock_llm_config):
    """Test that consolidation extracts important facts from episodic memories."""
    from klovis_agent.consolidation import ConsolidationEngine
    
    engine = ConsolidationEngine(llm_config=mock_llm_config)
    
    episodic_content = """
    Action: Created file hello.py
    Result: File contains print('Hello World')
    Verification: File exists and runs successfully
    """
    
    semantic_facts = await engine.consolidate(episodic_content)
    
    assert semantic_facts is not None
    assert len(semantic_facts) > 0
    # Should extract generalizable knowledge
    assert any("hello.py" in fact or "Hello World" in fact for fact in semantic_facts)

@pytest.mark.asyncio
async def test_consolidation_removes_temporal_details(temp_data_dir, mock_llm_config):
    """Test that consolidation removes time-specific details."""
    from klovis_agent.consolidation import ConsolidationEngine
    
    engine = ConsolidationEngine(llm_config=mock_llm_config)
    
    episodic_content = """
    At 2024-01-15 14:30:00, I executed code that printed today's date.
    The code ran in /tmp/workspace directory.
    """
    
    semantic_facts = await engine.consolidate(episodic_content)
    
    # Semantic facts should not contain specific timestamps or temp paths
    for fact in semantic_facts:
        assert "2024-01-15" not in fact
        assert "/tmp/" not in fact

@pytest.mark.asyncio
async def test_consolidation_handles_errors(temp_data_dir, mock_llm_config):
    """Test consolidation handles malformed input gracefully."""
    from klovis_agent.consolidation import ConsolidationEngine
    
    engine = ConsolidationEngine(llm_config=mock_llm_config)
    
    # Empty or malformed content
    semantic_facts = await engine.consolidate("")
    
    assert semantic_facts is not None
    assert isinstance(semantic_facts, list)
