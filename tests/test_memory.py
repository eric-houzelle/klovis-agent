import pytest
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

@pytest.mark.asyncio
async def test_episodic_memory_store_and_recall(temp_data_dir, mock_llm_config):
    """Test storing and recalling episodic memories."""
    from klovis_agent.memory.memory_manager import MemoryManager
    
    manager = MemoryManager(data_dir=temp_data_dir, ephemeral=True)
    await manager.initialize()
    
    # Store episodic memory
    memory_id = await manager.store_episodic(
        content="Test action: ran code execution",
        tags=["test", "code"]
    )
    
    assert memory_id is not None
    assert isinstance(memory_id, int)
    
    # Recall memory
    results = await manager.recall("code execution", zone="episodic", k=5)
    
    assert len(results) >= 1
    assert any("code execution" in r.get("content", "") for r in results)
    
    await manager.close()

@pytest.mark.asyncio
async def test_semantic_memory_persistence(temp_data_dir, mock_llm_config):
    """Test that semantic memories persist across sessions."""
    from klovis_agent.memory.memory_manager import MemoryManager
    
    # First session - store memory
    manager1 = MemoryManager(data_dir=temp_data_dir, ephemeral=False)
    await manager1.initialize()
    
    await manager1.store_semantic(
        content="Python is a programming language created by Guido van Rossum",
        tags=["programming", "python"]
    )
    await manager1.close()
    
    # Second session - recall memory
    manager2 = MemoryManager(data_dir=temp_data_dir, ephemeral=False)
    await manager2.initialize()
    
    results = await manager2.recall("Python programming language", zone="semantic", k=5)
    
    assert len(results) >= 1
    assert any("Python" in r.get("content", "") for r in results)
    
    await manager2.close()

@pytest.mark.asyncio
async def test_memory_ttl_pruning(temp_data_dir, mock_llm_config):
    """Test that old episodic memories are pruned based on TTL."""
    from klovis_agent.memory.memory_manager import MemoryManager
    
    manager = MemoryManager(data_dir=temp_data_dir, ephemeral=True, episodic_ttl_days=0)
    await manager.initialize()
    
    # Store memory
    await manager.store_episodic(content="Old action", tags=["old"])
    
    # Force pruning
    await manager.prune_expired()
    
    # Memory should be pruned
    results = await manager.recall("old action", zone="episodic")
    assert len(results) == 0
    
    await manager.close()

@pytest.mark.asyncio
async def test_memory_deduplication(temp_data_dir, mock_llm_config):
    """Test that duplicate semantic memories are merged."""
    from klovis_agent.memory.memory_manager import MemoryManager
    
    manager = MemoryManager(data_dir=temp_data_dir, ephemeral=True)
    await manager.initialize()
    
    # Store same content twice
    id1 = await manager.store_semantic(content="Unique fact about testing", tags=["test"])
    id2 = await manager.store_semantic(content="Unique fact about testing", tags=["test"])
    
    # Should return same ID or merge
    results = await manager.recall("Unique fact testing", zone="semantic", k=5)
    
    # Should only have one unique entry
    assert len(results) == 1
    
    await manager.close()
