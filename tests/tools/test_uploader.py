import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from buttermilk._core import StepRequest
from buttermilk.agents.evaluators.scorer import QualResults, QualScoreCRA
from buttermilk.agents.judge import AgentReasons
from buttermilk.utils.uploader import AsyncDataUploader
from buttermilk._core.config import SaveInfo
from buttermilk._core.contract import AgentOutput

# Sample data based on provided examples
SAMPLE_OUTPUTS = [
    AgentOutput(**{
                        "error": [],
                        "metadata": {},
                        "agent_id": "host-Yc8ekP",
                        "session_id": "20250424T0122Z-Rvon-c218d8dfd611-vscode",
                        "call_id": "Bc8scgvzT3vycJrrJUMhov",
                        "run_info": {
                            "platform": "local",
                            "name": "batch",
                            "job": "debugging",
                            "run_id": "20250424T0122Z-Rvon-c218d8dfd611-vscode",
                            "save_dir": "gs://prosocial-dev/runs/batch/debugging/20250424T0122Z-Rvon-c218d8dfd611-vscode",
                        },
                        "outputs": StepRequest(**{"role": "WAIT", "prompt": ""}),
                        "is_error": False
                    }),
    AgentOutput(**{
        "error": [],
        "metadata": {
            "finish_reason": "stop",
            "role": "judge",
            "name": "⚖️ Judge WRESDb"
        },
        "agent_id": "judge-WRESDb",
        "session_id": "20250424T0122Z-Rvon-c218d8dfd611-vscode",
        "call_id": "8MPyjSZt6PikCEMocsPFr6",
        "run_info": {
            "platform": "local",
            "name": "batch",
            "job": "debugging",
            "run_id": "20250424T0122Z-Rvon-c218d8dfd611-vscode",
        },
        "outputs": QualResults(**{
            "conclusion": "The content adheres to the guidelines.",
            "prediction": False,
            "confidence": "high"
        }),
        "is_error": False
    }),
    AgentOutput(**{
        "error": [],
        "metadata": {
            "role": "scorers",
            "name": "📊 Scorer MyVLKi"
        },
        "agent_id": "scorers-MyVLKi",
        "session_id": "20250424T0122Z-Rvon-c218d8dfd611-vscode",
        "call_id": "7R7w4Un76TDJaSzgp36gUW",
        "outputs": AgentReasons(**{
            "assessments": [QualScoreCRA(**{"correct": True, "feedback": "Feedback text"})]
        }),
        "is_error": False
    })
]

@pytest.fixture
def save_info(objs):
    """Create a test SaveInfo fixture."""
    return SaveInfo(**objs.save)
    

@pytest.fixture
def uploader(save_info):
    """Create a test AsyncDataUploader fixture."""
    return AsyncDataUploader(buffer_size=2, save_dest=save_info)

@pytest.mark.anyio
async def test_init(save_info):
    """Test that the uploader initializes correctly."""
    
    uploader = AsyncDataUploader(buffer_size=5, save_dest=save_info)
    
    assert uploader.buffer_size == 5
    assert uploader.save_dest == save_info
    assert uploader.buffer == []

@pytest.mark.anyio
async def test_add_single_item(uploader):
    """Test adding a single item below buffer threshold."""
    with patch.object(uploader, '_flush', AsyncMock()) as mock_flush:
        await uploader.add(SAMPLE_OUTPUTS[0])
        
        assert len(uploader.buffer) == 1
        assert uploader.buffer[0] == SAMPLE_OUTPUTS[0]
        mock_flush.assert_not_called()

@pytest.mark.anyio
async def test_add_triggers_flush(uploader):
    """Test adding items that trigger an automatic flush."""
    with patch.object(uploader, '_flush', AsyncMock()) as mock_flush:
        # Add enough items to trigger a flush (buffer_size=2)
        await uploader.add(SAMPLE_OUTPUTS[0])
        await uploader.add(SAMPLE_OUTPUTS[1])
        
        # Flush should have been called once
        mock_flush.assert_called_once()
        # Buffer should be empty after flush
        assert len(uploader.buffer) == 0

@pytest.mark.anyio
async def test_manual_flush(uploader):
    """Test manually flushing the buffer."""
    with patch.object(uploader, '_flush', AsyncMock()) as mock_flush:
        await uploader.add(SAMPLE_OUTPUTS[0])
        await uploader._flush()
        
        mock_flush.assert_called_once()
        assert len(uploader.buffer) == 0

@pytest.mark.anyio
async def test_empty_flush(uploader):
    """Test flushing an empty buffer."""
    with patch('buttermilk.utils.save.upload_rows_async', AsyncMock()) as mock_upload:
        await uploader._flush()
        
        mock_upload.assert_not_called()

@pytest.mark.anyio
async def test_flush_mechanics(save_info):
    """Test the actual flush mechanism with mocked upload function."""
    
    
    uploader = AsyncDataUploader(buffer_size=3, save_dest=save_info)
    
    with patch('buttermilk.utils.save.upload_rows_async', AsyncMock()) as mock_upload:
        # Add two items
        await uploader.add(SAMPLE_OUTPUTS[0])
        await uploader.add(SAMPLE_OUTPUTS[1])
        
        # Manually flush
        await uploader._flush()
        
        # Check that upload was called with correct parameters
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args[0]
        
        # First argument should be the list of items
        assert len(call_args[0]) == 2
        # Second argument should be the save_dest
        assert call_args[1] == save_info

@pytest.mark.anyio
async def test_aclose(uploader):
    """Test that aclose properly flushes remaining items."""
    with patch.object(uploader, 'flush', AsyncMock()) as mock_flush:
        await uploader.add(SAMPLE_OUTPUTS[0])
        await uploader.aclose()
        
        mock_flush.assert_called_once()

@pytest.mark.anyio
async def test_dict_conversion(uploader):
    """Test that items are properly converted to dictionaries before upload."""
    with patch('buttermilk.utils.save.upload_rows_async', AsyncMock()) as mock_upload:
        await uploader.add(SAMPLE_OUTPUTS[0])
        await uploader._flush()
        
        # Get the first argument passed to upload_rows_async (the list of dicts)
        uploaded_items = mock_upload.call_args[0][0]
        
        assert isinstance(uploaded_items[0], dict)
        assert uploaded_items[0]['agent_id'] == SAMPLE_OUTPUTS[0].agent_id
        assert uploaded_items[0]['session_id'] == SAMPLE_OUTPUTS[0].session_id


@pytest.mark.anyio
async def test_multiple_flushes(uploader):
    """Test multiple flushes with different items."""
    with patch('buttermilk.utils.save.upload_rows_async', AsyncMock()) as mock_upload:
        # First batch
        await uploader.add(SAMPLE_OUTPUTS[0])
        await uploader.add(SAMPLE_OUTPUTS[1])
        # Should trigger automatic flush
        
        # Second batch
        await uploader.add(SAMPLE_OUTPUTS[2])
        await uploader._flush()
        
        # Should have called upload twice
        assert mock_upload.call_count == 2
        
        # First call should have 2 items
        first_call_items = mock_upload.call_args_list[0][0][0]
        assert len(first_call_items) == 2
        
        # Second call should have 1 item
        second_call_items = mock_upload.call_args_list[1][0][0]
        assert len(second_call_items) == 1

@pytest.mark.anyio
async def test_error_handling_during_flush(save_info):
    """Test error handling during flush operation."""
    
    uploader = AsyncDataUploader(buffer_size=2, save_dest=save_info)
    
    # Mock upload_rows_async to raise an exception
    with patch('buttermilk.utils.save.upload_rows_async', 
               AsyncMock(side_effect=Exception("Upload failed"))) as mock_upload:
        with patch('buttermilk.utils.uploader.logger', MagicMock()) as mock_logger:
            # Add items and trigger flush
            await uploader.add(SAMPLE_OUTPUTS[0])
            await uploader.add(SAMPLE_OUTPUTS[1])
            
            # Verify logger.error was called
            mock_logger.error.assert_called()
            # Buffer should be cleared even if upload fails
            assert len(uploader.buffer) == 0

@pytest.mark.integration
@pytest.mark.anyio
async def test_integration_with_fixture_save(save_info):
    """Integration test using fixture objects to create SaveInfo."""
    # Create uploader with our fixture-created SaveInfo
    uploader = AsyncDataUploader(buffer_size=2, save_dest=save_info)
    
    # Test basic functionality
    await uploader.add(SAMPLE_OUTPUTS[0])
    await uploader.add(SAMPLE_OUTPUTS[1])  # This should trigger a flush
        
    # todo: check the rows were actually inserted
    pass