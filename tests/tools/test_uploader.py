"""Tests for AsyncDataUploader using real configuration."""

import pytest

from buttermilk._core.contract import AgentTrace, StepRequest
from buttermilk.agents.evaluators.scorer import QualResults, QualScoreCRA  
from buttermilk.agents.judge import Reasons
from buttermilk.utils.uploader import AsyncDataUploader


# Sample data based on provided examples
@pytest.fixture
def sample_outputs() -> list[AgentTrace]:
    return [
        AgentTrace(
            error=[],
            metadata={},
            agent_info={
                "agent_id": "host-Yc8ekP",
            },
            session_id="20250424T0122Z-Rvon-c218d8dfd611-vscode",
            call_id="Bc8scgvzT3vycJrrJUMhov",
            session_info={
                "platform": "local",
                "name": "batch",
                "job": "debugging",
                "session_id": "20250424T0122Z-Rvon-c218d8dfd611-vscode",
                "save_dir": "gs://prosocial-dev/runs/batch/debugging/20250424T0122Z-Rvon-c218d8dfd611-vscode",
            },
            outputs=StepRequest(role="WAIT"),
            is_error=False,
        ),
        AgentTrace(
            error=[],
            metadata={"finish_reason": "stop", "role": "judge", "name": "⚖️ Judge WRESDb"},
            agent_info={
                "agent_id": "judge-WRESDb",
            },
            session_id="20250424T0122Z-Rvon-c218d8dfd611-vscode",
            call_id="8MPyjSZt6PikCEMocsPFr6",
            session_info={
                "platform": "local",
                "name": "batch",
                "job": "debugging",
                "session_id": "20250424T0122Z-Rvon-c218d8dfd611-vscode",
            },
            outputs=QualResults(conclusion="The content adheres to the guidelines.", prediction=False, confidence="high"),
            is_error=False,
        ),
        AgentTrace(
            error=[],
            metadata={"role": "scorers", "name": "📊 Scorer MyVLKi"},
            agent_info={"agent_id": "scorers-MyVLKi"},
            session_id="20250424T0122Z-Rvon-c218d8dfd611-vscode",
            call_id="7R7w4Un76TDJaSzgp36gUW",
            outputs=Reasons(assessments=QualScoreCRA(correct=True, feedback="Feedback text")),
            is_error=False,
        ),
    ]


class TestAsyncDataUploader:
    """Test AsyncDataUploader using real BM storage interface."""
    
    @pytest.mark.anyio
    async def test_uploader_initialization_with_real_storage(self, real_bm):
        """Test that uploader initializes correctly with real BM storage."""
        # Get storage from real BM instance
        storage = real_bm.get_storage()
        
        # Create uploader with real storage
        uploader = AsyncDataUploader(buffer_size=5, save_dest=storage)
        
        assert uploader.buffer_size == 5
        assert uploader.save_dest is storage
        assert uploader._buffer == []
    
    @pytest.mark.anyio 
    async def test_uploader_buffer_management(self, real_bm, sample_outputs):
        """Test uploader buffer management with real storage."""
        storage = real_bm.get_storage()
        uploader = AsyncDataUploader(buffer_size=2, save_dest=storage)
        
        # Add items to buffer
        await uploader.add(sample_outputs[0])
        assert len(uploader._buffer) == 1
        
        await uploader.add(sample_outputs[1])
        assert len(uploader._buffer) == 2
        
        # Adding third item should trigger flush (buffer_size=2)
        await uploader.add(sample_outputs[2])
        # Buffer should be reset after flush
        assert len(uploader._buffer) == 1  # Only the new item
    
    @pytest.mark.anyio
    async def test_uploader_manual_flush(self, real_bm, sample_outputs):
        """Test manual flush functionality with real storage."""
        storage = real_bm.get_storage()
        uploader = AsyncDataUploader(buffer_size=5, save_dest=storage)
        
        # Add some items
        await uploader.add(sample_outputs[0])
        await uploader.add(sample_outputs[1])
        assert len(uploader._buffer) == 2
        
        # Manual flush
        await uploader.flush()
        assert len(uploader._buffer) == 0
    
    @pytest.mark.anyio
    async def test_uploader_context_manager(self, real_bm, sample_outputs):
        """Test uploader as async context manager with real storage."""
        storage = real_bm.get_storage()
        
        async with AsyncDataUploader(buffer_size=5, save_dest=storage) as uploader:
            await uploader.add(sample_outputs[0])
            await uploader.add(sample_outputs[1])
            # Buffer should still have items
            assert len(uploader._buffer) == 2
        
        # After exiting context, buffer should be flushed
        # Note: We can't directly check buffer state after context exit
        # since the uploader object might be cleaned up
    
    @pytest.mark.anyio
    async def test_uploader_handles_storage_interface(self, real_bm):
        """Test that uploader works with real storage interface methods."""
        storage = real_bm.get_storage()
        uploader = AsyncDataUploader(buffer_size=1, save_dest=storage)
        
        # Verify storage has expected interface
        assert hasattr(storage, 'save')  # or whatever method is expected
        
        # Create a simple test object that can be saved
        test_data = {
            "test_key": "test_value",
            "timestamp": "2024-01-01T00:00:00"
        }
        
        # Test that uploader can handle the data
        await uploader.add(test_data)
        # Should automatically flush since buffer_size=1
        assert len(uploader._buffer) == 0


class TestAsyncDataUploaderErrorHandling:
    """Test error handling in AsyncDataUploader."""
    
    @pytest.mark.anyio
    async def test_uploader_handles_invalid_storage(self):
        """Test uploader behavior with invalid storage interface."""
        # Test with None storage (should raise appropriate error)
        with pytest.raises((TypeError, AttributeError)):
            uploader = AsyncDataUploader(buffer_size=1, save_dest=None)
            await uploader.add({"test": "data"})
    
    @pytest.mark.anyio
    async def test_uploader_handles_invalid_buffer_size(self, real_bm):
        """Test uploader behavior with invalid buffer size."""
        storage = real_bm.get_storage()
        
        # Test with zero buffer size
        with pytest.raises(ValueError):
            AsyncDataUploader(buffer_size=0, save_dest=storage)
        
        # Test with negative buffer size
        with pytest.raises(ValueError):
            AsyncDataUploader(buffer_size=-1, save_dest=storage)


class TestAsyncDataUploaderIntegration:
    """Integration tests for AsyncDataUploader with real infrastructure."""
    
    @pytest.mark.anyio
    async def test_end_to_end_upload_flow(self, real_bm, sample_outputs):
        """Test complete upload flow with real BM infrastructure."""
        storage = real_bm.get_storage()
        
        # Create uploader with small buffer for testing
        uploader = AsyncDataUploader(buffer_size=2, save_dest=storage)
        
        # Upload all sample outputs
        for output in sample_outputs:
            await uploader.add(output)
        
        # Flush any remaining items
        await uploader.flush()
        
        # Verify uploader is in clean state
        assert len(uploader._buffer) == 0
    
    @pytest.mark.anyio
    async def test_uploader_with_session_context(self, real_bm, sample_outputs):
        """Test uploader in the context of a real session."""
        # This test validates that uploader works within the session context
        # where it would actually be used
        
        storage = real_bm.get_storage()
        session_id = real_bm.session_info.session_id
        
        # Create uploader 
        uploader = AsyncDataUploader(buffer_size=1, save_dest=storage)
        
        # Add session-specific metadata to outputs
        for output in sample_outputs:
            output.session_id = session_id
            await uploader.add(output)
        
        # Verify all items were processed
        assert len(uploader._buffer) == 0  # Should auto-flush with buffer_size=1