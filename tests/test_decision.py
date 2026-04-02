import pytest
from unittest.mock import MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_decision_with_high_confidence(mock_llm_config, temp_data_dir):
    """Test decision making when confidence is high."""
    from klovis_agent.decision import DecisionEngine
    from klovis_agent.models.decision import DecisionRequest
    
    engine = DecisionEngine(llm_config=mock_llm_config)
    
    request = DecisionRequest(
        goal="Complete the task",
        context="Previous actions succeeded",
        available_actions=["search", "code_execution", "file_read"]
    )
    
    # Mock LLM response with high confidence decision
    decision = await engine.decide(request)
    
    assert decision is not None
    assert decision.action in ["search", "code_execution", "file_read", "terminate"]
    assert 0 <= decision.confidence <= 1

@pytest.mark.asyncio
async def test_decision_termination_when_goal_complete(mock_llm_config, temp_data_dir):
    """Test that agent terminates when goal is complete."""
    from klovis_agent.decision import DecisionEngine
    from klovis_agent.models.decision import DecisionRequest
    
    engine = DecisionEngine(llm_config=mock_llm_config)
    
    request = DecisionRequest(
        goal="Write hello.py",
        context="File hello.py has been created and verified",
        available_actions=["search", "code_execution", "file_read"],
        goal_complete=True
    )
    
    decision = await engine.decide(request)
    
    assert decision.action == "terminate"

@pytest.mark.asyncio
async def test_decision_replan_on_failure(mock_llm_config, temp_data_dir):
    """Test that agent replans when actions fail."""
    from klovis_agent.decision import DecisionEngine
    from klovis_agent.models.decision import DecisionRequest
    
    engine = DecisionEngine(llm_config=mock_llm_config)
    
    request = DecisionRequest(
        goal="Execute code",
        context="Last action failed with error: timeout",
        available_actions=["search", "code_execution"],
        consecutive_failures=2
    )
    
    decision = await engine.decide(request)
    
    # Should either try different action or replan
    assert decision is not None
    assert decision.confidence < 0.8  # Lower confidence on failures
