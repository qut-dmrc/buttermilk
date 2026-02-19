"""Tests for session storage integration in AutogenOrchestrator.

This test verifies that AutogenOrchestrator properly wires SessionStorageService
to log batch session messages during groupchat execution.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from buttermilk._core.types import RunRequest
from buttermilk.orchestrators.groupchat import AutogenOrchestrator


class TestAutogenOrchestratorSessionStorage:
    """Test session storage integration in AutogenOrchestrator."""

    @pytest.mark.anyio
    async def test_orchestrator_calls_finalize_session_after_run_completes(self, real_bm):
        """Test that AutogenOrchestrator calls SessionStorageService.finalize_session() after run completes.

        This test should FAIL because AutogenOrchestrator currently does not:
        1. Instantiate SessionStorageService
        2. Call finalize_session() in the finally block

        Once implemented, the orchestrator should call finalize_session() with
        the session_id when the run completes (success or failure).
        """
        # Create minimal orchestrator config
        orchestrator_config = {
            "agents": {},  # No agents needed for this test
            "observers": {},
            "parameters": {},
        }

        # Create orchestrator instance
        orchestrator = AutogenOrchestrator(**orchestrator_config)

        # Mock SessionStorageService to verify it gets called
        mock_storage = Mock()
        mock_storage.finalize_session = Mock()

        # Create a minimal RunRequest
        session_id = "test-session-123"
        request = RunRequest(
            flow="test-flow",  # Required field
            session_id=session_id,
            inputs={},
            callback_to_ui=AsyncMock(),  # Provide mock callback to avoid warnings
        )

        # Patch SessionStorageService where it's used (imported into groupchat module)
        with patch(
            "buttermilk.orchestrators.groupchat.SessionStorageService",
            return_value=mock_storage,
        ):
            # Patch _setup to avoid full initialization but still let finally block run
            mock_termination = Mock()
            mock_termination.has_terminated = True
            mock_interrupt = Mock()

            with patch.object(orchestrator, "_setup", return_value=(mock_termination, mock_interrupt)):
                # Run the orchestrator
                # The finally block should call finalize_session
                try:
                    await orchestrator._run(request)
                except Exception:
                    # We expect errors since we're mocking heavily
                    pass

        # Verify finalize_session was called with the session_id
        # This assertion will FAIL because:
        # 1. SessionStorageService is not imported in groupchat.py
        # 2. _storage_service attribute is not created
        # 3. finalize_session() is not called in the finally block
        assert mock_storage.finalize_session.called, "finalize_session was not called - SessionStorageService not wired in AutogenOrchestrator"

        if mock_storage.finalize_session.called:
            call_args = mock_storage.finalize_session.call_args
            assert call_args[0][0] == session_id, f"Expected session_id {session_id}, got {call_args[0][0]}"
            assert call_args[0][1] in [
                "completed",
                "failed",
            ], f"Expected status 'completed' or 'failed', got {call_args[0][1]}"
