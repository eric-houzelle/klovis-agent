import pytest
from datetime import datetime

@pytest.mark.asyncio
async def test_result_aggregation(temp_data_dir, mock_llm_config):
    """Test that agent results are properly aggregated."""
    from klovis_agent.result import ResultAggregator
    from klovis_agent.models.result import ActionResult
    
    aggregator = ResultAggregator()
    
    # Add multiple actions
    aggregator.add_action(ActionResult(
        action="code_execution",
        result={"success": True, "output": "Hello"},
        timestamp=datetime.now()
    ))
    
    aggregator.add_action(ActionResult(
        action="file_write",
        result={"success": True, "path": "test.py"},
        timestamp=datetime.now()
    ))
    
    # Generate summary
    summary = aggregator.generate_summary()
    
    assert summary is not None
    assert "code_execution" in summary or "file_write" in summary
    assert len(aggregator.actions) == 2

@pytest.mark.asyncio
async def test_result_with_artifacts(temp_data_dir, mock_llm_config):
    """Test result includes generated artifacts."""
    from klovis_agent.result import ResultAggregator
    from klovis_agent.models.result import ActionResult
    
    aggregator = ResultAggregator()
    
    aggregator.add_action(ActionResult(
        action="file_write",
        result={"success": True, "path": "output.txt", "size": 1024},
        timestamp=datetime.now(),
        artifacts=["output.txt"]
    ))
    
    final_result = aggregator.finalize(goal="Create output file")
    
    assert final_result.artifacts is not None
    assert "output.txt" in final_result.artifacts

@pytest.mark.asyncio
async def test_result_failure_handling(temp_data_dir, mock_llm_config):
    """Test result handles action failures gracefully."""
    from klovis_agent.result import ResultAggregator
    from klovis_agent.models.result import ActionResult
    
    aggregator = ResultAggregator()
    
    aggregator.add_action(ActionResult(
        action="code_execution",
        result={"success": False, "error": "Timeout"},
        timestamp=datetime.now(),
        error="Execution timeout"
    ))
    
    summary = aggregator.generate_summary()
    
    assert summary is not None
    assert "failed" in summary.lower() or "error" in summary.lower()
