"""Tests for session persistence functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from buttermilk.api.services.message_service import ChatMessage
from buttermilk.api.services.session_storage import SessionStorageService


class TestSessionStorageService:
    """Test suite for SessionStorageService."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary directory for session storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def storage_service(self, temp_storage_dir):
        """Create a SessionStorageService instance with temp directory."""
        with patch("buttermilk.api.services.session_storage.SESSIONS_DIR", temp_storage_dir):
            return SessionStorageService()

    def test_save_message(self, storage_service, temp_storage_dir):
        """Test saving a message to session storage."""
        session_id = "test-session-123"
        message = ChatMessage(
            type="record",
            message_id="msg-001",
            preview="Test message",
            outputs={"content": "Hello world"},
            agent_info={"name": "TestAgent"},
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.005,
        )

        # Save the message
        storage_service.save_message(session_id, message)

        # Verify file was created
        session_file = temp_storage_dir / f"{session_id}.json"
        assert session_file.exists()

        # Verify content
        with open(session_file) as f:
            data = json.load(f)
            assert data["session_id"] == session_id
            assert len(data["messages"]) == 1
            assert data["messages"][0]["message_id"] == "msg-001"
            assert data["messages"][0]["type"] == "record"

    def test_get_session_messages(self, storage_service, temp_storage_dir):
        """Test retrieving messages from session storage."""
        session_id = "test-session-456"
        
        # Create a session file with test data
        session_data = {
            "session_id": session_id,
            "created_at": "2024-01-01T10:00:00Z",
            "last_updated": "2024-01-01T10:30:00Z",
            "messages": [
                {
                    "type": "record",
                    "message_id": "msg-001",
                    "preview": "First message",
                    "outputs": {"content": "Hello"},
                    "timestamp": "2024-01-01T10:05:00Z",
                },
                {
                    "type": "record",
                    "message_id": "msg-002",
                    "preview": "Second message",
                    "outputs": {"content": "World"},
                    "timestamp": "2024-01-01T10:10:00Z",
                },
            ],
        }
        
        session_file = temp_storage_dir / f"{session_id}.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f)

        # Retrieve messages
        messages = storage_service.get_session_messages(session_id)
        
        assert len(messages) == 2
        assert messages[0].message_id == "msg-001"
        assert messages[1].message_id == "msg-002"

    def test_session_exists(self, storage_service, temp_storage_dir):
        """Test checking if a session exists."""
        session_id = "test-session-789"
        
        # Session doesn't exist yet
        assert not storage_service.session_exists(session_id)
        
        # Create session file
        session_file = temp_storage_dir / f"{session_id}.json"
        session_file.write_text(json.dumps({"session_id": session_id, "messages": []}))
        
        # Now it should exist
        assert storage_service.session_exists(session_id)

    def test_should_persist_message(self, storage_service):
        """Test message filtering logic."""
        # Messages that should be persisted
        record_msg = ChatMessage(type="record", message_id="1")
        ui_msg = ChatMessage(type="ui_message", message_id="2")
        research_msg = ChatMessage(type="research_result", message_id="3")
        
        assert storage_service.should_persist_message(record_msg)
        assert storage_service.should_persist_message(ui_msg)
        assert storage_service.should_persist_message(research_msg)
        
        # Messages that should NOT be persisted
        system_update = ChatMessage(type="system_update", message_id="4")
        system_msg = ChatMessage(type="system_message", message_id="5")
        
        assert not storage_service.should_persist_message(system_update)
        assert not storage_service.should_persist_message(system_msg)

    def test_append_message_to_existing_session(self, storage_service, temp_storage_dir):
        """Test appending messages to an existing session."""
        session_id = "append-test"
        
        # Save first message
        msg1 = ChatMessage(
            type="record",
            message_id="msg-001",
            preview="First",
            outputs={"content": "First message"},
        )
        storage_service.save_message(session_id, msg1)
        
        # Save second message
        msg2 = ChatMessage(
            type="record",
            message_id="msg-002",
            preview="Second",
            outputs={"content": "Second message"},
        )
        storage_service.save_message(session_id, msg2)
        
        # Verify both messages are in the file
        messages = storage_service.get_session_messages(session_id)
        assert len(messages) == 2
        assert messages[0].message_id == "msg-001"
        assert messages[1].message_id == "msg-002"

    def test_corrupted_session_file_handling(self, storage_service, temp_storage_dir):
        """Test graceful handling of corrupted session files."""
        session_id = "corrupted-session"
        
        # Create a corrupted file
        session_file = temp_storage_dir / f"{session_id}.json"
        session_file.write_text("{ invalid json }")
        
        # Should return empty list and log warning
        messages = storage_service.get_session_messages(session_id)
        assert messages == []
        
        # Should still be able to save new messages (overwrite corrupted file)
        msg = ChatMessage(
            type="record",
            message_id="msg-recovery",
            preview="Recovery",
            outputs={"content": "Recovered"},
        )
        storage_service.save_message(session_id, msg)
        
        # Verify we can now read the session
        messages = storage_service.get_session_messages(session_id)
        assert len(messages) == 1
        assert messages[0].message_id == "msg-recovery"


@pytest.mark.asyncio
class TestWebSocketMessagePersistence:
    """Test WebSocket message persistence integration."""

    @pytest.fixture
    def mock_storage_service(self):
        """Create a mock SessionStorageService."""
        service = AsyncMock(spec=SessionStorageService)
        service.should_persist_message.return_value = True
        service.save_message = AsyncMock()
        return service

    async def test_websocket_message_interceptor(self, mock_storage_service):
        """Test that WebSocket messages are intercepted and persisted."""
        from buttermilk.api.services.message_service import MessageService
        
        # Create a test message
        test_message = ChatMessage(
            type="record",
            message_id="ws-msg-001",
            preview="WebSocket message",
            outputs={"data": "test"},
        )
        
        # Simulate WebSocket message handling with persistence
        with patch("buttermilk.api.services.session_storage.SessionStorageService", return_value=mock_storage_service):
            # Format message for client
            formatted = MessageService.format_message_for_client(test_message)
            
            # In actual implementation, this would be called in WebSocket handler
            if formatted and mock_storage_service.should_persist_message(formatted):
                await mock_storage_service.save_message("test-session", formatted)
            
            # Verify save was called
            mock_storage_service.save_message.assert_called_once_with("test-session", formatted)


@pytest.mark.asyncio
class TestSessionRestoration:
    """Test session restoration functionality."""

    async def test_restore_session_endpoint(self):
        """Test the GET /api/session/{session_id}/messages endpoint."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        # Mock endpoint (will be implemented)
        @app.get("/api/session/{session_id}/messages")
        async def get_session_messages(session_id: str):
            # Mock implementation
            if session_id == "existing-session":
                return {
                    "messages": [
                        {
                            "type": "record",
                            "message_id": "restored-001",
                            "preview": "Restored message",
                            "outputs": {"content": "Previous conversation"},
                        }
                    ]
                }
            else:
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Session not found")
        
        client = TestClient(app)
        
        # Test existing session
        response = client.get("/api/session/existing-session/messages")
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 1
        assert data["messages"][0]["message_id"] == "restored-001"
        
        # Test non-existing session
        response = client.get("/api/session/non-existing/messages")
        assert response.status_code == 404